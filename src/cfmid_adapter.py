import subprocess
import re
import math
import time
import os
import shutil
from typing import List, Tuple, Dict, Any
from config import INSTANCE_ID, POOL_DIR_BASE

CURRENT_WORKER_ID = None 

class CFMIDAdapter:
    def __init__(self, docker_image="wishartlab/cfmid:latest"):
        self.docker_image = docker_image
        self.spectrum_cache: Dict[str, List[Dict[str, Any]]] = {} 
        
        base_path = "/trained_models_cfmid4.0/cfmid4"

        self.adduct_models = {
            "[M+H]+": {
                "param": f"{base_path}/[M+H]+/param_output.log",
                "config": f"{base_path}/[M+H]+/param_config.txt"
            },
            "[M-H]-": {
                "param": f"{base_path}/[M-H]-/param_output.log",
                "config": f"{base_path}/[M-H]-/param_config.txt"
            }
        }
        
        self.adduct_models["[M+Na]+"] = self.adduct_models["[M+H]+"]
        self.adduct_models["[M]+"] = self.adduct_models["[M+H]+"]
        self.adduct_models["M+H"] = self.adduct_models["[M+H]+"]
        self.adduct_models["M+Na"] = self.adduct_models["[M+H]+"]
        self.adduct_models["M-H"] = self.adduct_models["[M-H]-"]

        self.has_timeout = False

    @staticmethod
    def prepare_pool(num_workers, cpus, mem, base_dir, image):
        pool_dir = os.path.join(base_dir, INSTANCE_ID)
        print(f"🚀 Initializing Docker Pool for Instance '{INSTANCE_ID}' ({num_workers} containers)...")
        
        if os.path.exists(pool_dir):
            try: shutil.rmtree(pool_dir)
            except: pass
        os.makedirs(pool_dir, exist_ok=True)
        
        subprocess.run(f"docker rm -f $(docker ps -a -q --filter name=cfmid_worker_{INSTANCE_ID}_)", shell=True, stderr=subprocess.DEVNULL)

        for i in range(num_workers):
            worker_dir = os.path.join(pool_dir, f"w_{i}")
            worker_tmp_dir = os.path.join(worker_dir, "tmp_files")
            os.makedirs(worker_dir, exist_ok=True)
            os.makedirs(worker_tmp_dir, exist_ok=True)
            
            container_name = f"cfmid_worker_{INSTANCE_ID}_{i}"
            
            cmd = [
                "docker", "run", "-d",
                f"--name={container_name}",
                f"--cpus={cpus}", f"--memory={mem}",
                "-v", f"{worker_dir}:/data",
                "-v", f"{worker_tmp_dir}:/tmp", 
                "-w", "/data",  
                image,
                "sleep", "infinity"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"✅ Docker Pool '{INSTANCE_ID}' Ready.")

    @staticmethod
    def cleanup_pool():
        print(f"🧹 Cleaning up Docker Pool for '{INSTANCE_ID}'...")
        subprocess.run(f"docker rm -f $(docker ps -a -q --filter name=cfmid_worker_{INSTANCE_ID}_)", shell=True, stderr=subprocess.DEVNULL)

    def reset_status(self):
        self.has_timeout = False

    def _get_container_name(self):
        if CURRENT_WORKER_ID is None:
             return f"cfmid_worker_{INSTANCE_ID}_0"
        return f"cfmid_worker_{INSTANCE_ID}_{CURRENT_WORKER_ID}"

    def _parse_cfmid_output(self, output_str: str) -> List[Tuple[float, float]]:
        peaks_dict = {}
        lines = output_str.split('\n')
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit(): continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    mz = float(parts[0])
                    inten = float(parts[1])
                    if inten <= 0: continue 
                    mz_key = round(mz, 4) 
                    if mz_key not in peaks_dict or inten > peaks_dict[mz_key]:
                        peaks_dict[mz_key] = inten
            except: continue
        return [(mz, inten) for mz, inten in peaks_dict.items()]

    def _generate_fraggraph(self, smiles: str, ionization_mode: str = "+") -> Dict[float, str]:
        fragment_map = {}
        container_name = self._get_container_name()
        
        depth = 1
        output_filename = "frag_d1.txt"
        
        subprocess.run(["docker", "exec", container_name, "rm", "-f", output_filename], 
                       stderr=subprocess.DEVNULL)

        internal_timeout = 300
        python_timeout = internal_timeout + 15
        
        cmd_gen = [
            "docker", "exec",
            "-e", "OMP_NUM_THREADS=1",
            "-e", "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib",
            container_name,
            "/opt/cfm/bin/fraggraph-gen", 
            smiles, str(depth), ionization_mode, "fullgraph", output_filename
        ]
        
        try:
            subprocess.run(cmd_gen, capture_output=True, timeout=python_timeout, check=True)
            
            cmd_read = ["docker", "exec", container_name, "cat", output_filename]
            res = subprocess.run(cmd_read, capture_output=True, text=True, check=True)
            
            content = res.stdout
            if not content.strip():
                return {}
            
            for line in content.split('\n'):
                if line.startswith("#") or not line.strip(): continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        mass = float(parts[1])
                        frag_smiles = parts[2]
                        if mass not in fragment_map or len(frag_smiles) > len(fragment_map[mass]):
                            fragment_map[mass] = frag_smiles
                    except: continue
                    
        except Exception:
            return {}
                
        return fragment_map

    def predict_spectrum(self, smiles: str, adduct: str = "[M]+", retries: int = 3) -> List[Dict[str, Any]]:
        if not smiles: return []
        
        cache_key = f"{smiles}_{adduct}"
        if cache_key in self.spectrum_cache:
            return self.spectrum_cache[cache_key]

        model_paths = self.adduct_models.get(adduct, self.adduct_models["[M+H]+"])
        container_name = self._get_container_name()
        
        internal_timeout = 1800
        python_timeout = internal_timeout + 10
        
        cmd = [
            "docker", "exec",
            "-e", "OMP_NUM_THREADS=1",
            "-e", "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib",
            container_name,
            "/opt/cfm/bin/cfm-predict",
            smiles, "0.05", 
            model_paths["param"], model_paths["config"],
            "1", "stdout", "0"
        ]
        
        raw_peaks = []
        is_success = False
        
        for attempt in range(retries):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=python_timeout, check=True)
                raw_peaks = self._parse_cfmid_output(result.stdout)
                is_success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"\n[CFM-ID Error] Worker: {self._get_container_name()}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Return Code: {e.returncode}")
                print(f"Stderr: {e.stderr}") 
                
                if e.returncode in [124, 137]: continue
                continue
            except subprocess.TimeoutExpired:
                print(f"\n[CFM-ID Timeout] Worker: {self._get_container_name()}")
                continue 
            except Exception as e: 
                print(f"\n[CFM-ID Unknown Error] {e}")
                time.sleep(0.5)
                continue
            
        if not is_success:
            self.has_timeout = True
            return []
        
        if not raw_peaks: return []

        try:
            max_int = max([p[1] for p in raw_peaks])
        except ValueError: return []

        if max_int <= 0: return []
        
        simple_peaks = []
        for mz, inten in raw_peaks:
            rel_int = (inten / max_int) * 100.0
            if rel_int < 1.0: continue
            simple_peaks.append({
                "mz": mz,
                "intensity": rel_int
            })

        self.spectrum_cache[cache_key] = simple_peaks
        return simple_peaks
    
    def annotate_peaks(self, smiles: str, peaks: List[Dict[str, Any]], adduct: str = "[M]+") -> List[Dict[str, Any]]:
        if not peaks: return []
        ion_mode = "-" if "-" in adduct else "+"
        
        frag_map = self._generate_fraggraph(smiles, ionization_mode=ion_mode)
        
        annotated_peaks = []
        tolerance = 0.1
        
        for p in peaks:
            new_p = p.copy()
            mz = p['mz']
            matched_smiles = "Unknown"
            
            best_diff = float('inf')
            if frag_map:
                for f_mass, f_smi in frag_map.items():
                    diff = abs(f_mass - mz)
                    if diff < tolerance and diff < best_diff:
                        best_diff = diff
                        matched_smiles = f_smi
            
            new_p['smiles'] = matched_smiles
            annotated_peaks.append(new_p)
            
        return annotated_peaks

    def compute_similarity(self, pred_peaks: List[Dict[str, Any]], target_peaks: List[Tuple[float, float]], tolerance=0.5) -> float:
        if not pred_peaks or not target_peaks: return 0.0
        simple_pred = [(p['mz'], p['intensity']) for p in pred_peaks]
        
        def bin_spectrum(peaks):
            bins = {}
            max_i = max([p[1] for p in peaks]) if peaks else 1.0
            for mz, i in peaks:
                b = int(mz / tolerance)
                norm_i = math.sqrt((i / max_i) * 100.0)
                bins[b] = max(bins.get(b, 0), norm_i)
            return bins

        vec_p = bin_spectrum(simple_pred)
        vec_t = bin_spectrum(target_peaks)
        
        all_bins = set(vec_p.keys()) | set(vec_t.keys())
        dot = sum(vec_p.get(b,0)*vec_t.get(b,0) for b in all_bins)
        norm_p = sum(v**2 for v in vec_p.values())
        norm_t = sum(v**2 for v in vec_t.values())
        
        if norm_p == 0 or norm_t == 0: return 0.0
        return dot / (math.sqrt(norm_p) * math.sqrt(norm_t))
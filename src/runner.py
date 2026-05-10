import json
import logging
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil
from tqdm import tqdm

from cfmid_adapter import CFMIDAdapter
from checkpoint import append_result, load_checkpoint, summarize_final_results
from data_loader import MolecularDataLoader
from worker import init_worker, process_single_sample

logger = logging.getLogger(__name__)
global_executor = None
active_config = None


def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=str(log_level).upper(),
        format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
    )


def register_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def kill_all_children():
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        if children:
            for child in children:
                try:
                    child.kill()
                except Exception:
                    logger.warning("Failed to kill child process pid=%s", child.pid, exc_info=True)
            _, alive = psutil.wait_procs(children, timeout=2)
            if alive:
                logger.warning("Some child processes did not exit after kill: %s", [p.pid for p in alive])
    except Exception:
        logger.exception("Failed while killing child processes")


def signal_handler(signum, frame):
    global global_executor
    if global_executor:
        try:
            global_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            logger.warning("Executor does not support cancel_futures; falling back to shutdown(wait=False)")
            global_executor.shutdown(wait=False)

    if active_config:
        CFMIDAdapter.cleanup_pool(active_config.docker.instance_id)
    kill_all_children()
    os._exit(1)


def build_tasks(df):
    tasks = []
    for idx, row in df.iterrows():
        import selfies as sf

        raw_smi = row.get("smiles", "")
        raw_sf = row.get("selfies", "")
        true_smi = ""

        if raw_smi and isinstance(raw_smi, str):
            true_smi = raw_smi
        elif raw_sf and isinstance(raw_sf, str):
            try:
                true_smi = sf.decoder(raw_sf)
            except Exception:
                logger.warning("Failed to decode SELFIES for sample=%s", idx, exc_info=True)

        task = row.to_dict()
        task["index"] = idx
        task["true_smiles"] = true_smi

        if isinstance(row.get("fps"), str):
            task["target_fp_list"] = json.loads(row["fps"])
        else:
            task["target_fp_list"] = row.get("fps", [])

        p_raw = row.get("peaks")
        if isinstance(p_raw, str):
            p_list = json.loads(p_raw)
        else:
            p_list = p_raw

        cleaned = []
        if p_list:
            for p in p_list:
                try:
                    if float(p[1]) > 0:
                        cleaned.append((float(p[0]), float(p[1])))
                except Exception:
                    logger.warning("Skipping malformed peak for sample=%s peak=%r", idx, p, exc_info=True)

        task["target_peaks_cleaned"] = cleaned
        tasks.append(task)

    return tasks


def run_experiment(config):
    global global_executor, active_config
    active_config = config

    output_dir = os.path.dirname(config.paths.output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    loader = MolecularDataLoader()
    df = loader.load_test_data(config.paths.test_file_path, limit=config.run.max_test_samples)
    tasks = build_tasks(df)

    tasks, running_metrics = load_checkpoint(config.paths.output_file_path, tasks, logger)
    if not tasks:
        print("All tasks completed.")
        return

    max_workers = config.run.pworkers
    print(f"Starting execution with {max_workers} workers (Mode: {config.run.search_mode})...")

    try:
        CFMIDAdapter.prepare_pool(
            num_workers=max_workers,
            cpus=config.docker.cpus,
            mem=config.docker.mem,
            base_dir=config.docker.pool_dir_base,
            image=config.docker.image,
            instance_id=config.docker.instance_id,
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize Docker pool: {e}")
        CFMIDAdapter.cleanup_pool(config.docker.instance_id)
        return

    manager = None
    try:
        manager = multiprocessing.Manager()
        id_queue = manager.Queue()
        for i in range(max_workers):
            id_queue.put(i)
    except Exception as e:
        print(f"FATAL: Failed to create Manager Queue: {e}")
        CFMIDAdapter.cleanup_pool(config.docker.instance_id)
        return

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(id_queue, config),
        ) as pool:
            global_executor = pool
            future_to_idx = {pool.submit(process_single_sample, t): t["index"] for t in tasks}
            pbar = tqdm(
                as_completed(future_to_idx),
                total=len(tasks),
                desc="Processing",
                mininterval=0.5,
            )

            for future in pbar:
                try:
                    res = future.result()
                except Exception:
                    failed_idx = future_to_idx[future]
                    logger.exception("Worker future failed for sample=%s", failed_idx)
                    continue

                if not res:
                    continue

                append_result(config.paths.output_file_path, res)
                running_metrics.update_from_result(res)
                averages = running_metrics.averages()

                status = str(res.get("mcts_status", "N/A"))
                display_status = status[:25] + ".." if len(status) > 25 else status
                pbar.set_postfix_str(
                    f"T1:{averages['top1']:.4f} "
                    f"TK:{averages['topk']:.4f} "
                    f"VNR:{averages['valid_node_rate']:.2f} "
                    f"dR:{averages['avg_reward_delta']:.3f} | {display_status}"
                )

    finally:
        print("Cleaning up resources...")
        global_executor = None
        CFMIDAdapter.cleanup_pool(config.docker.instance_id)
        if manager:
            try:
                manager.shutdown()
            except Exception:
                logger.exception("Failed to shut down multiprocessing manager")

    summarize_final_results(config.paths.output_file_path, config.run.search_mode, logger)

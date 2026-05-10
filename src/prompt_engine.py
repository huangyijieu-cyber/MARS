from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import logging
from rdkit import Chem 
from rdkit.Chem import AllChem
from fingerprint_utils import explain_shared_bits 

logger = logging.getLogger(__name__)

def get_top_peaks_string(peaks_list, top_k=5):
    if not peaks_list: return "N/A"
    try:
        cleaned_peaks = [(float(p[0]), float(p[1])) for p in peaks_list]
        sorted_peaks = sorted(cleaned_peaks, key=lambda x: x[1], reverse=True)
        formatted_peaks = [f"{p[0]:.1f} ({p[1]:.1f}%)" for p in sorted_peaks[:top_k]]
        return ", ".join(formatted_peaks)
    except Exception:
        logger.warning("Failed to format target peaks; falling back to raw string: %r", peaks_list, exc_info=True)
        return str(peaks_list)

def build_molrag_prompt(top_k=10):
    template = """
Role Definition:
You are an expert in mass spectrometry interpretation and chemical structure reconstruction. Your task is to deduce the most probable
target molecular structure based on reference molecules (References) and a target molecular formula ({target_formula}),
and convert it into a SMILES sequence.
I. Core Task:
The system provides reference molecules retrieved based on spectral similarity. You need to adopt different inference strategies to
construct the target molecule depending on the Similarity Score of the reference molecules.
II. Task Input:
1. Reference Molecules (References):
{context}
2. Target Molecular Formula: {target_formula}
3. Target Mass Spectrum Key Peaks (Target Top Peaks m/z), peaks given in the format m/z (relative intensity %):
{target_top_peaks}
III. Chain-of-Thought (CoT) Guidelines
Please strictly follow the steps below for your deduction STEP BY STEP:
Step 1: Reference Evaluation and Strategy Selection
First, check the Similarity scores of all reference molecules in the Input:
[Case A: High-Confidence References Exist (At least one molecule with Similarity ≥ 0.6)]
- 1. Observe all reference molecules (References) with Similarity greater than 0.6 to identify their common structural features
(e.g., benzene rings, heterocycles, long chains, specific functional groups).
- 2. Check if the Target Top Peaks (m/z) support the core scaffold of the reference molecules. If strong peaks appear in the target
spectrum that cannot be explained by the reference scaffold, or if characteristic fragmentation peaks expected from the reference
scaffold are missing, determine this as a “Feature Conflict”. As a mass spectrometry expert, you must decisively switch to Case
B.
- 3. Combining with Target Top Peaks m/z (fragment peaks), if the m/z suggests specific functional groups not present in the
reference molecules (such as carbonyl, hydroxyl, halogen), please prioritize marking them.
- 4. Determine the core scaffold that the target molecule very likely possesses based on the SMILES sequences of the reference
molecules.
- Note: You must compare the common features of similar molecules with the features corresponding to the Target Top Peaks (m/z)
for any contradictions. If there is a conflict, you must decisively switch to Case B!!!! Do not be deceived by the high-similarity
of the reference molecules!!!
[Case B: Lack of Credible References (All molecules with Similarity < 0.6)]
- 1. Abandon the scaffold: The value of the reference molecules is extremely low; treat their Shared Substructures only as potential
fragment hints (not as a fixed scaffold).
- 2. De Novo Inference: You must rely primarily on the {target_formula} and Target Top Peaks (m/z) for inference.
- 3. Basis:
– Calculate the Degree of Unsaturation (DoU) of the {target_formula}.
– Use Target Top Peaks (m/z) for fragment inference: Analyze the correspondence between strong peaks m/z and common
fragment ions (e.g., 91→benzyl, 77→phenyl, 43→acetyl/propyl, 57→butyl, 29→ethyl); Check for reasonable Neutral
Loss (e.g., M-18 dehydration, M-15 demethylation).
– Construct several candidate isomers that fit the molecular formula and can explain the main fragments.
- 4. Skip the gap analysis directly and proceed to “Step 3” for structure reconstruction.
Step 2: Atom Gap and Unsaturation Analysis (Gap Analysis) — Only applicable to [Case A]
- 1. Calculate Atom Difference: {target_formula} - Core Scaffold Formula = Remaining Atoms.
- 2. Calculate Unsaturation Difference (DoU Diff): Decrease in H (May indicate formation of double-bonds, rings, or introduction
of C=O); Increase in H (May indicate double-bond reduction or ring opening).
- 3. Key Decision: Judge the form in which the remaining atoms exist to match the substitution sites of the scaffold based on
Shared Substructures and Target Top Peaks.
Step 3: Structure Reconstruction & Validation
- 1. Execute Construction: If [Case A]: Add deduced side chains or functional groups to reasonable positions on the core scaffold.
If [Case B]: Combine deduced functional groups and carbon chains to generate a reasonable molecule that fits the target formula.
- 2. Chemical Rationality Check: Ensure the constructed molecule is chemically stable under standard conditions; Prioritize stable
functional group combinations common in nature or synthetic chemistry.
- 3. Self-Correction — The most important step: Atom Count (Strictly match the element types and quantities of the target formula); Syntax Check (Ensure the output SMILES sequence is syntactically closed and valid); Valence Check.
IV. Output Requirements
1. Please output the Reasoning first following the CoT order. In the Reasoning, explicitly state whether you fall under [Case A] or
[Case B] and proceed with the deduction accordingly.
2. Output the final result in JSON format. The result list should contain {top_k} candidate SMILES, sorted from high to low
compliance, with the sequence best fitting the task requirements ranked first.
Reasoning:
Please strictly follow the provided CoT output reasoning process.
[Your CoT here...]
Final Answer (JSON), strictly as a JSON list of strings:
JSON
{{
“SMILES STRING 1”,
“SMILES STRING 2”,
...
}}
"""
    return ChatPromptTemplate.from_template(template).partial(top_k=str(top_k))

def build_unified_mcts_prompt():
    template = """
Role Definition:
You are a Mass Spectrometry Forensics Expert & Synthetic Chemist. Your goal is to evolve a molecular structure to perfectly
match a Target Spectrum, strictly adhering to the Target Formula.
I. Core Task:
The system provides a diagnostic report based on the target spectrum and the predicted spectrum of the current molecule. You need
to adopt different inference strategies based on the discrepancies between the two spectra to modify or reconstruct the current
molecule.
II. Task Inputs:
1. Target Formula: {formula} (The exact atomic composition you must strictly match)
2. Current Structure: {smiles} (The molecule you generated in the previous step)
3. Spectral Discrepancy (Diagnosis):
This section compares the Target Spectrum (Ground-Truth) vs. Predicted Spectrum (Current Structure).
- MATCHED: Peaks present in both. The SMILES shown is the fragment structure generated by the current molecule. (Action:
Preserve these substructures)
- EXTRA: Peaks present ONLY in your prediction. The SMILES shown is the fragment you incorrectly generated. (Action:
Modify the molecule to REMOVE this fragment.)
- MISSING: Peaks present ONLY in the Target. (Action: You need to ADD a substructure that produces this mass.)
[Detailed Report]:
{spectrum_comparison}
4. Reference Knowledge Hints (From RAG):
(Structural clues retrieved from external database based on mass similarity)
{reference_hints}
(CRITICAL: If a reference molecule has a peak we are missing, try to borrow its substructure!)
III. Chain-of-Thought (CoT) Guidelines:
Please strictly follow the steps below for your deduction STEP BY STEP:
Step 1: Fundamental Analysis
- 1. Calculate Degree of Unsaturation (DoU) for the {formula}.
- 2. Check if the Current Structure matches this DoU. If not, ring/double-bond adjustment is mandatory.
Step 2: Strategy Selection (Case Decision)
- Evaluate the severity of the Spectral Discrepancy:
- [Case A: Optimization]: The Base Peak (strongest peak) matches, but minor fragments are missing/extra. → Strategy: Perform
local modifications (e.g., move a functional group, isomerize a side chain, heteroatom swap).
- [Case B: Reconstruction]: The Base Peak is MISSING or the fragmentation pattern implies a wrong core scaffold. → Strategy:
Abandon the current scaffold. Propose a DIFFERENT isomer (e.g., open/close rings, change from linear to branched, change
ring size) that explains the missing major fragments.
Step 3: Fragment Mapping & Execution
- 1. Map the “Missing Peaks” (m/z) to specific substructures (e.g., 91→Benzyl, 77→Phenyl, 43→Acetyl).
- 2. Apply the modification determined in Step 2.
- FINAL CHECK: Count atoms. The output SMILES MUST have exactly {formula}.
IV. Output Requirements:
1. Reasoning: Explicitly state “Decision: Case A” or “Decision: Case B”, then explain your chemical logic.
2. Final Answer: A JSON object containing the SINGLE BEST modified SMILES string.
Reasoning:
Please strictly follow the provided CoT output reasoning process.
[Your CoT here...]
Final Answer (JSON), strictly as a JSON list of strings:
JSON
{{
“SMILES”: “YOUR MODIFIED SMILES”
}}
"""
    return ChatPromptTemplate.from_template(template)

def format_docs_for_context(docs, target_fp_array=None):
    formatted_str = ""
    target_bits_set = set()
    if target_fp_array is not None:
        if isinstance(target_fp_array, list):
            target_bits_set = set(np.where(np.array(target_fp_array) > 0)[0])
        elif isinstance(target_fp_array, np.ndarray):
            target_bits_set = set(np.where(target_fp_array > 0)[0])

    for i, doc in enumerate(docs):
        shared_substructures_str = "N/A" 
        score = doc.metadata.get('score', 0)
        smiles = doc.metadata.get('smiles', '')
        
        if smiles and len(target_bits_set) > 0:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
                    retrieved_bits_set = set(fp_vect.GetOnBits())
                    shared_bits = sorted(list(target_bits_set.intersection(retrieved_bits_set)))
                    frags = explain_shared_bits(mol, shared_bits)
                    if frags:
                        shared_substructures_str = ", ".join(list(set(frags)))
            except Exception:
                logger.warning("Failed to explain shared fingerprint bits for doc=%s smiles=%r", i + 1, smiles, exc_info=True)
        
        formatted_str += (
            f"Reference {i+1} (Similarity: {score:.4f}):\n" 
            f"  - Formula: {doc.metadata.get('formula', 'N/A')}\n"
            f"  - SMILES: {smiles}\n"
            f"  - Shared Substructures (Matched Fingerprints): {shared_substructures_str}\n"
            "-------------------\n"
        )
    return formatted_str

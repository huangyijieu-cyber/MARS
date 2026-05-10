import os
from dataclasses import dataclass

import pandas as pd


def is_valid_status(status_str):
    s = str(status_str)
    if "Timeout" in s:
        return False
    if any(x in s for x in ["Init Failed", "Skipped", "Error"]):
        return False
    return True


@dataclass
class RunningMetrics:
    valid_finished_count: int = 0
    sum_top1_acc: float = 0.0
    sum_topk_acc: float = 0.0
    sum_valid_node_rate: float = 0.0
    sum_avg_reward_delta: float = 0.0

    def update_from_result(self, result):
        status = str(result.get("mcts_status", "N/A"))
        if not is_valid_status(status):
            return

        self.valid_finished_count += 1
        self.sum_top1_acc += result.get("top1_mol_accuracy", 0)
        self.sum_topk_acc += result.get("topk_mol_accuracy", 0)
        self.sum_valid_node_rate += float(result.get("valid_node_rate", 0.0))
        self.sum_avg_reward_delta += float(result.get("avg_reward_delta", 0.0))

    def averages(self):
        if self.valid_finished_count <= 0:
            return {
                "top1": 0.0,
                "topk": 0.0,
                "valid_node_rate": 0.0,
                "avg_reward_delta": 0.0,
            }

        return {
            "top1": self.sum_top1_acc / self.valid_finished_count,
            "topk": self.sum_topk_acc / self.valid_finished_count,
            "valid_node_rate": self.sum_valid_node_rate / self.valid_finished_count,
            "avg_reward_delta": self.sum_avg_reward_delta / self.valid_finished_count,
        }


def load_checkpoint(output_file_path, tasks, logger):
    metrics = RunningMetrics()
    finished_indices = set()

    if not os.path.exists(output_file_path):
        print("-> No checkpoint found. Starting fresh.")
        return tasks, metrics

    try:
        print(f"Checking for existing checkpoint: {output_file_path}")
        existing_df = pd.read_csv(output_file_path)
        if existing_df.empty or "index" not in existing_df.columns:
            print("-> Checkpoint empty or invalid. Starting fresh.")
            return tasks, metrics

        finished_indices = set(existing_df["index"].astype(int).unique())
        print(f"-> Found {len(finished_indices)} finished samples.")

        for _, row in existing_df.iterrows():
            metrics.update_from_result(row)

        original_len = len(tasks)
        remaining_tasks = [t for t in tasks if t["index"] not in finished_indices]
        print(f"-> Resuming execution. Remaining tasks: {len(remaining_tasks)} / {original_len}")
        return remaining_tasks, metrics

    except Exception:
        logger.exception("Error reading checkpoint; starting fresh from %s", output_file_path)
        return tasks, metrics


def append_result(output_file_path, result):
    single_df = pd.DataFrame([result])
    header_needed = not os.path.exists(output_file_path)
    single_df.to_csv(output_file_path, mode="a", header=header_needed, index=False)


def summarize_final_results(output_file_path, search_mode, logger):
    try:
        final_df = pd.read_csv(output_file_path)
        print(f"\nExperiment Finished. Total Samples: {len(final_df)}")

        if "mcts_status" not in final_df.columns:
            return

        valid_df = final_df[final_df["mcts_status"].apply(is_valid_status)]

        if valid_df.empty:
            print("No valid samples found for metrics calculation.")
            return

        top1_agg = {
            "top1_valid_mols": valid_df["top1_valid_mols"].mean(),
            "top1_mces_dist": valid_df["top1_mces_dist"].mean(),
            "top1_tanimoto_sim": valid_df["top1_tanimoto_sim"].mean(),
            "top1_mol_accuracy": valid_df["top1_mol_accuracy"].mean(),
        }
        topk_agg = {
            "topk_valid_mols": valid_df["topk_valid_mols"].mean(),
            "topk_mces_dist": valid_df["topk_mces_dist"].mean(),
            "topk_tanimoto_sim": valid_df["topk_tanimoto_sim"].mean(),
            "topk_mol_accuracy": valid_df["topk_mol_accuracy"].mean(),
        }
        process_agg = {
            "valid_node_rate": valid_df["valid_node_rate"].mean() if "valid_node_rate" in valid_df.columns else 0.0,
            "avg_reward_delta": valid_df["avg_reward_delta"].mean() if "avg_reward_delta" in valid_df.columns else 0.0,
        }

        def fmt(d):
            return {k: round(float(v), 4) for k, v in d.items()}

        print("\n" + "=" * 40)
        print(f"Final Metrics (Valid Samples: {len(valid_df)}, Mode: {search_mode})")
        print("=" * 40)
        print(f"Process: {fmt(process_agg)}")
        print(f"Top1: {fmt(top1_agg)}")
        print(f"Topk: {fmt(topk_agg)}")

    except Exception:
        logger.exception("Could not load final results for summary from %s", output_file_path)

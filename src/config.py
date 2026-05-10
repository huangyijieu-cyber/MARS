import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import yaml


DEFAULT_CONFIG_PATH = "configs/molpuzzle.yaml"


@dataclass
class ApiConfig:
    model: str
    base_url: str
    api_key_env: List[str]
    api_key: Optional[str]
    temperature: float


@dataclass
class PathConfig:
    faiss_db_dir: str
    test_file_path: str
    output_file_path: str


@dataclass
class RunConfig:
    search_mode: int
    max_test_samples: Optional[int]
    pworkers: int
    log_level: str


@dataclass
class RetrievalConfig:
    retrieval_k: int
    context_k: int
    fingerprint_dim: int


@dataclass
class GenerationConfig:
    top_k: int
    initial_max_retries: int


@dataclass
class MCTSConfig:
    iterations: int


@dataclass
class DockerConfig:
    image: str
    cpus: float
    mem: str
    pool_dir_base: str
    instance_id_env: str
    default_instance_id: str
    instance_id: str


@dataclass
class AppConfig:
    api: ApiConfig
    paths: PathConfig
    run: RunConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    mcts: MCTSConfig
    docker: DockerConfig
    config_path: str


def parse_args():
    parser = argparse.ArgumentParser(description="Run MARS molecular structure elucidation evaluation.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    api_raw = raw.get("api", {})
    paths_raw = raw.get("paths", {})
    run_raw = raw.get("run", {})
    retrieval_raw = raw.get("retrieval", {})
    generation_raw = raw.get("generation", {})
    mcts_raw = raw.get("mcts", {})
    docker_raw = raw.get("docker", {})

    api_key_env = list(api_raw.get("api_key_env", ["MARS_API_KEY", "OPENAI_API_KEY"]))
    api_key = _first_env_value(api_key_env)

    instance_id_env = docker_raw.get("instance_id_env", "INSTANCE_ID")
    default_instance_id = docker_raw.get("default_instance_id", "default")
    instance_id = os.getenv(instance_id_env, default_instance_id)

    return AppConfig(
        api=ApiConfig(
            model=api_raw.get("model", "deepseek-chat"),
            base_url=api_raw.get("base_url", "https://api.deepseek.com"),
            api_key_env=api_key_env,
            api_key=api_key,
            temperature=api_raw.get("temperature", 0),
        ),
        paths=PathConfig(
            faiss_db_dir=paths_raw.get("faiss_db_dir", "data/faiss_kb"),
            test_file_path=paths_raw.get("test_file_path", "data/MolPuzzle_threshold_0.2.tsv"),
            output_file_path=paths_raw.get("output_file_path", "results/MARS.csv"),
        ),
        run=RunConfig(
            search_mode=run_raw.get("search_mode", 3),
            max_test_samples=run_raw.get("max_test_samples"),
            pworkers=run_raw.get("pworkers", 1),
            log_level=run_raw.get("log_level", "INFO"),
        ),
        retrieval=RetrievalConfig(
            retrieval_k=retrieval_raw.get("retrieval_k", 20),
            context_k=retrieval_raw.get("context_k", 5),
            fingerprint_dim=retrieval_raw.get("fingerprint_dim", 4096),
        ),
        generation=GenerationConfig(
            top_k=generation_raw.get("top_k", 10),
            initial_max_retries=generation_raw.get("initial_max_retries", 3),
        ),
        mcts=MCTSConfig(
            iterations=mcts_raw.get("iterations", 5),
        ),
        docker=DockerConfig(
            image=docker_raw.get("image", "wishartlab/cfmid:latest"),
            cpus=docker_raw.get("cpus", 1.5),
            mem=docker_raw.get("mem", "5g"),
            pool_dir_base=docker_raw.get("pool_dir_base", "/tmp/cfmid_pool_data"),
            instance_id_env=instance_id_env,
            default_instance_id=default_instance_id,
            instance_id=instance_id,
        ),
        config_path=path,
    )


def _first_env_value(names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None

import os
import multiprocessing

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from runner import register_signal_handlers, run_experiment, setup_logging

SEARCH_MODE = 3


def main():
    setup_logging()
    register_signal_handlers()
    run_experiment(search_mode=SEARCH_MODE)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

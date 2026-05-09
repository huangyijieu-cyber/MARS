import os

# --- API Configuration ---
MODEL = os.getenv("MARS_MODEL", "deepseek-chat")        # Model
API_KEY = os.getenv("MARS_API_KEY") or os.getenv("OPENAI_API_KEY")  # API key
BASE_URL = os.getenv("MARS_BASE_URL", "https://api.deepseek.com")  # API base URL
TEMPERATURE = 0                                         # Temperature

# --- Path Configuration ---
FAISS_DB_DIR = "data/faiss_kb"                         # Knowledge base path
TEST_FILE_PATH = "data/MolPuzzle_threshold_0.2.tsv"    # Test data path
OUTPUT_FILE_PATH = "results/MARS.csv"                  # Output result path

# --- Parameter Configuration ---
MCTS_ITERATIONS = 5               # Number of MCTS iterations
RETRIEVAL_K = 20                  # Size of the retrieval molecule pool; CONTEXT_K molecules are selected from it for prompting
CONTEXT_K = 5                     # Number of context molecules
FINGERPRINT_DIM = 4096            # Dimension of molecular fingerprint
TOP_K = 10                        # Number of Top-K results for final output
MAX_TEST_SAMPLES = 2              # Test sample limit, set to None for unlimited
PWORKERS = 1                      # Number of parallel workers, adjust based on CPU cores

# === Docker Container Pool Configuration (New) ===
DOCKER_CPUS = 1.5                                    # Number of CPUs allocated per Docker container
DOCKER_MEM = "5g"                                    # Memory size allocated per Docker container
POOL_DIR_BASE = "/tmp/cfmid_pool_data"               # Base path for Docker container pool data directory

# Instance ID, used to distinguish different run tasks
# Default is 'default', can be overridden via environment variable
INSTANCE_ID = os.getenv("INSTANCE_ID", "default")

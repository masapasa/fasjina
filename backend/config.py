# indexing data
DOCARRAY_PULL_NAME = 'fashion-multimodal-all'
DATA_DIR = "/home/aswin/data/archive/images/" # Where are the files?
CSV_FILE = "/home/aswin/data/archive/styles.csv" # Where's the metadata?
WORKSPACE_DIR = "../embeddings"
MAX_DOCS = 99
DEVICE = "cpu"

# PQLiteIndexer
DIMS = 512 # This should be same shape as vector embedding

# serving via REST
# HOST = "http://0.0.0.0"
# PORT = 12346

# metas for executors
TIMEOUT_READY = -1 # Wait forever for executor to be ready. Good for slow connections

# cloud
# CLOUD_HOST = "https://30b1c61a65.wolf.jina.ai"

# CLOUD_HOST = "grpcs://fash-2ef056cc3d.wolf.jina.ai"
IMAGE_ROOT_URL = "https://examples.jina.ai/data/fashion/images/" # where are the image files?

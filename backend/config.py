# indexing data
DATA_DIR = "/home/aswin/data/archive/images/" # Where are the files?
CSV_FILE = "/home/aswin/data/archive/styles.csv" # Where's the metadata?
MAX_DOCS = 11
WORKSPACE_DIR = "workspace"
DEVICE = "cpu"

# PQLiteIndexer
COLUMNS = [
    ("gender", "str"),
    ("masterCategory", "str"),
    ("subCategory", "str"),
    ("articleType", "str"),
    ("baseColour", "str"),
    ("season", "str"),
    ("usage", "str"),
    ("year", "int"),
]
DIMS = 512 # This should be same shape as vector embedding

# searching via gRPC
search_terms = ("Dress", "Shirt", "Shoe")

# serving via REST
SERVER = "0.0.0.0" # remove http://
PORT = 12345

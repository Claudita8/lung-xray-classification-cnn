from src.utils.config import load_config
from src.utils.seed import set_seed

if __name__ == "__main__":
    config = load_config("configs/cnn_baseline.yaml")
    set_seed(config["split"]["seed"])

    print("Project:", config["project"]["name"])
    print("Classes:", config["data"]["class_names"])

import yaml
import os

def load_config(config_path="configs/embedding.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def cleaned_data_path(config, epoch):
    return config["paths"]["cleaned_data_template"].format(epoch=epoch)

def embeddings_output_path(config, epoch, model_key):
    return config["paths"]["embeddings_output_template"].format(epoch=epoch, model_key=model_key)
import yaml

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)

def save_config(config: Dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f)
from pprint import pprint

from pipeline.config import load_config
from pipeline.paths import ensure_data_dirs


def main() -> None:
    config = load_config()
    print("Loaded pipeline config:")
    pprint(config)
    print("Data directories:")
    for name, path in ensure_data_dirs(config.project_root).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

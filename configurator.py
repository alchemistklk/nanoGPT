import sys
import yaml
import argparse
import train


def load_config(config_file):
    with open(config_file, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


def override_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = load_config(config_file)
    parser = argparse.ArgumentParser(description="Override config with command line arguments")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size")
    parser.add_argument("--bias", type=bool, default=True, help="Whether to use bias in the model")
    parser.add_argument("--real_data", type=bool, default=True, help="Whether to use real data")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type to use")
    parser.add_argument("--compile", type=bool, default=True, help="Whether to compile the model")
    parser.add_argument("--profiling", type=bool, default=False, help="Whether to profile the model")
    
    args = parser.parse_args(sys.argv[2:])
    config = override_config_with_args(config, args)
    train(config)
    
    
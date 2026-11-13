"""Training entry point for SpikeSort-Net."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.parse_args()


if __name__ == "__main__":
    main()

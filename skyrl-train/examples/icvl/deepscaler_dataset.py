"""
Preprocess the DeepScaleR-Preview-Dataset to parquet format for ICVL training.

Dataset: https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset
The dataset contains ~40k math problems with answers from AIME, AMC, Omni-MATH, etc.

Usage:
    uv run examples/icvl_deepscaler/deepscaler_dataset.py --output_dir $HOME/data/deepscaler
"""

import argparse
import os

import datasets


INSTRUCTION = (
    "Please solve the following math problem step by step. "
    "Put your final answer within \\boxed{}."
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess DeepScaleR dataset")
    parser.add_argument("--output_dir", default="~/data/deepscaler")
    parser.add_argument("--val_ratio", type=float, default=0.01,
                        help="Fraction of training data to hold out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    print(f"Loading dataset: {data_source}")
    ds = datasets.load_dataset(data_source, split="train")
    print(f"Total examples: {len(ds)}")

    # Split into train/val
    split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    def make_map_fn(split_name):
        def process_fn(example, idx):
            problem = example["problem"]
            answer = example["answer"]

            prompt_text = f"{problem}\n\n{INSTRUCTION}"

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                "env_class": "aime",
                "reward_model": {
                    "ground_truth": answer,
                    "style": "rule",
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "problem": problem,
                    "answer": answer,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Saved train ({len(train_dataset)} examples) to: {train_path}")
    print(f"Saved val ({len(val_dataset)} examples) to: {val_path}")


if __name__ == "__main__":
    main()

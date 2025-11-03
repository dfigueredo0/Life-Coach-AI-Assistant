from pathlib import Path
import argparse

from src.model.pretokenizer.pretokenize import pretokenize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=Path, required=True)
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-val", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--min-assistant-tokens", type=int, default=25)
    args = ap.parse_args()
    pretokenize(args.trees, args.out_train, args.out_val, args.val_ratio, args.min_assistant_tokens)

if __name__ == "__main__":
    main()
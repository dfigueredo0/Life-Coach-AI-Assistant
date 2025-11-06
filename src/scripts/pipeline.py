# scripts/pipeline.py
from src.data.synth_seed import main as synth_main
from src.data.make_dataset import main as make_main
from src.data.split_sft import main as split_main

def main():
    synth_main()
    import sys
    sys.argv = [
        "make_dataset",
        "--raw", "data/raw_synth.jsonl",
        "--out-sft", "data/sft.jsonl",
        "--out-ppo", "data/ppo.jsonl",
    ]
    make_main()
    split_main()

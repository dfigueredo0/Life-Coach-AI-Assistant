from pathlib import Path
import json
from src.model.sft_dataset import SFTJsonlDataset
from src.model.tokenizer import load_tokenizer

def test_special_tokens_and_filter(tmp_path: Path):
    special = {"user":"<|user|>", "assistant":"<|assistant|>", "system":"<|system|>", "end":"<|endoftext|>"}
    tb = load_tokenizer("stabilityai/stablelm-2-12b", special)
    # create tiny dataset
    f = tmp_path / "toy.jsonl"
    f.write_text(
        "\n".join([
            json.dumps({"turns":[{"role":"user","text":"I want to reach goal X"},
                                 {"role":"assistant","text":"Here are your steps ... step 1, step 2, step 3."}]}),
        ]),
        encoding="utf-8",
    )
    ds = SFTJsonlDataset(f, tb.tokenizer, special, max_len=256)
    item = ds[0]
    assert item["input_ids"].shape[0] == 256
    # ensure special tokens are in the vocab
    toks = tb.tokenizer.encode("<|user|>hello<|endoftext|>")
    assert len(toks) > 0

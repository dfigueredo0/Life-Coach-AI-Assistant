from __future__ import annotations
import json, gzip

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

DEFAULT_SPECIAL = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "end": "<|endoftext|>",
}

def _open_any(path: Path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')

def iter_trees(trees_file: Path) -> Iterator[Dict]:
    with _open_any(trees_file) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                
def flatten_tree_to_turns(tree: Dict) -> List[Dict]:
    """
        Converts a single converstaion tree into a simple sequence of turns. 
        Keep only linear paths from root to each leaf producing multiple samples per tree

    Args:
        tree (Dict): _description_

    Returns:
        List[Dict]: _description_
    """
    id2node = {n['message_id']: n for n in tree['messages']}
    root_ids = [m["message_id"] for m in tree["messages"] if m.get("parent_id") is None]
    
    paths = []
    
    def dfs(mid: str, acc: List[Dict]):
        node = id2node[mid]
        role = 'user' if node['role'] == 'prompter' else 'assistant'
        acc2 = acc + [{'role': role, 'text': node['text']}]
        childs = [m['message_id'] for m in tree['messages'] if m.get('parent_id') == mid]
        if not childs:
            paths.append(acc2)
        else:
            for c in childs:
                dfs(c, acc2)
    
    for r in root_ids:
        dfs(r, [])
    
    return [{'turns': p} for p in paths]

def filter_tag(record: Dict, min_assistant_tokens: int = 25) -> Optional[Dict]:
    """
        Enforce basic quality filter: at least one assistant turn with >= min tokens.
        Token approximation: split by whitespace    

    Args:
        record (Dict): _description_
        min_assistant_tokens (int, optional): _description_. Defaults to 25.

    Returns:
        Optional[Dict]: _description_
    """
    
    max_asst_len = 0
    for t in record['turns']:
        if t['role'] == 'assistant':
            max_asst_len = max(max_asst_len, len(t['text'].split()))
    if max_asst_len < min_assistant_tokens:
        return None
    return record

def format_stf(record: Dict, special: Dict[str, str] = DEFAULT_SPECIAL) -> Dict:
    """
        Output JSONL with 'turns' preserved.

    Args:
        record (Dict): _description_
        special (Dict[str, str], optional): _description_. Defaults to DEFAULT_SPECIAL.

    Returns:
        Dict: _description_
    """
    
    #TODO: Add domain metadata here (energy/location tags)
    return record

def pretokenize(trees_file: Path, out_train: Path, out_val: Path, val_ratio: float = 0.02, min_assistant_tokens: int = 25, special: Dict[str, str] = DEFAULT_SPECIAL):
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    
    train_f = open(out_train, 'w', encoding='utf-8')
    val_f = open(out_val, 'w', encoding='utf-8')
    
    for i, tree in enumerate(iter_trees(trees_file)):
        for rec in flatten_tree_to_turns(tree):
            rec = filter_tag(rec, min_assistant_tokens)
            if not rec:
                continue
            rec = format_stf(rec, special)
            target = val_f if (i % int(1/val_ratio) == 0) else train_f
            target.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    train_f.close()
    val_f.close()
    
def convert_domain_pairs(src_jsonl: Path, out_jsonl: Path, include_energy: bool = True, include_location: bool = True):
    """ Convert your domain pairs:
      { "goal": "...", "plan": "...", "energy":"low", "location":"home" }
    to dialog turns expected by SFT:
      turns: [ {role:"user", text:"I want to ..."}, {role:"assistant", text:"Here are your steps ..."} ]

    Args:
        src_jsonl (Path): _description_
        out_jsonl (Path): _description_
        include_energy (bool, optional): _description_. Defaults to True.
        include_location (bool, optional): _description_. Defaults to True.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(src_jsonl, 'r', encoding='utf-8') as f, open(out_jsonl, 'w', encoding='utf-8') as g:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            user_txt = rec['goal']
            plan_txt = rec['plan']
            
            if include_energy and rec.get('energy'):
                plan_txt += f"\n\n[energy:{rec['energy']}]"
            if include_location and rec.get('location'):
                plan_txt += f"\n\n[location:{rec['location']}]"
                
            out = {"turns": [{"role": "user", "text": user_txt}, {"role": "assistant", "text": plan_txt}]}
            g.write(json.dumps(out, ensure_ascii=False) + "\n")
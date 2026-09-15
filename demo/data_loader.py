"""Public replay loader: synthetic fixtures only, no original news or frozen outputs."""
import json
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'samples'

def load_replay_cases(data_dir=DATA_DIR):
    rows = [json.loads(line) for line in (data_dir / 'example_input.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(row.get('synthetic') is not True for row in rows):
        raise ValueError('Public replay requires explicitly synthetic fixtures')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate fixture ID')
    return {row['id']: row for row in rows}

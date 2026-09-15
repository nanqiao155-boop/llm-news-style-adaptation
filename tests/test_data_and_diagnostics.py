from pathlib import Path
import pytest
from src.training.sft_dataset_access import resolve_sft_file,gradient_training_files,DatasetAccessError
from scripts.lora_v2_production_decode import repetition_diagnostics

def test_split_guard(tmp_path):
    for name in ('train','validation','test'):
        (tmp_path/(name+'.jsonl')).write_text('{}\n')
    assert gradient_training_files(tmp_path)==(tmp_path/'train.jsonl',)
    with pytest.raises(DatasetAccessError):resolve_sft_file(tmp_path,'final_evaluation')
    assert resolve_sft_file(tmp_path,'model_selection_evaluation').name=='validation.jsonl'

def test_severe_repetition_boundary():
    rows=[{'prediction':'\n'.join(['虚构测试句。']*n)} for n in (2,3,4,5)]
    metrics=repetition_diagnostics(rows)
    assert metrics['sample_count']==4
    assert metrics['any_line_repeat_ge_3']==3
    assert metrics['severe_repeat_ge_5']==1

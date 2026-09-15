import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r'.')
INTERIM = ROOT/'data/interim'
PROCESSED = ROOT/'data/processed'
sys.path.insert(0,str(PROJECT))

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    sys.modules[name]=m
    s.loader.exec_module(m)
    return m

base=module('frozen_generation',r'experiments/lightweight/private_workspace\validation24_20260907\run_validation24.py')
sha=base.sha256_file
atomic=base.atomic_json

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []

def jsonl(path,items):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
    os.replace(tmp,path)

def export(name,value):
    atomic(PROCESSED/name,value)
    atomic(ROOT/name,value)

def export_rows(name,value):
    jsonl(PROCESSED/name,value); jsonl(ROOT/name,value)

def csvout(name,items):
    for path in (PROCESSED/name,ROOT/name):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]) if items else ['status'])
            w.writeheader(); w.writerows(items)

def inputs():
    p=INTERIM/'validation24.jsonl'
    if p.exists(): return rows(p)
    pack,selected=base.load_inputs()
    assert all(r.get('split','validation')=='validation' for r in selected)
    jsonl(p,selected)
    atomic(INTERIM/'validation_provenance.json',{'pack':str(base.PACK),'pack_sha256':sha(base.PACK),'validation_source':str(base.VALIDATION),'validation_sha256':sha(base.VALIDATION),'sample_ids':pack['sample_ids'],'count':24,'split':'validation','test_accessed':False})
    return selected

CONDITIONS=['lfm_f16','lfm_ptq','lfm_qad','qwen_q4_lora']
MODELS={k:ROOT/'models'/n for k,n in zip(CONDITIONS[:3],['LFM2.5-1.2B-Instruct-F16.gguf','LFM2.5-1.2B-Instruct-Q4_0.gguf','LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf'])}
MODELS['qwen_q4_lora']=Path(r'experiments/lightweight/private_workspace\precision_sweep_bartowski_same_source_20260907\models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf')
ADAPTER=base.ADAPTER
SERVER=base.SERVER
PORT=18135
LFM_SAMPLING={'temperature':0.1,'top_k':50,'top_p':0.95,'min_p':0.05,'repeat_penalty':1.05,'repeat_last_n':64,'seed':42,'presence_penalty':0.0,'frequency_penalty':0.0}

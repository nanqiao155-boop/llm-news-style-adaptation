from common import *
import argparse
import re
import subprocess
import threading
import time
import urllib.request
import psutil

def get(path):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}'+path,timeout=5) as r:return json.load(r)

def gpu():
    out={}
    for key,args in [('memory',['--query-gpu=name,memory.used','--format=csv,noheader,nounits']),('compute_apps',['--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])]:
        p=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        out[key]=p.stdout.strip();out[key+'_exit']=p.returncode
    return out

class Monitor:
    def __init__(self,p):
        self.p=psutil.Process(p.pid); self.samples=[];self.done=threading.Event();self.phase='load'
        self.t=threading.Thread(target=self.loop,daemon=True);self.t.start()
    def sample(self):
        try:
            m=self.p.memory_info();v=psutil.virtual_memory()
            self.samples.append({'time':time.time(),'phase':self.phase,'working_set_mib':m.rss/2**20,'private_mib':m.private/2**20,'os_peak_working_set_mib':m.peak_wset/2**20,'os_peak_pagefile_mib':m.peak_pagefile/2**20,'system_available_gib':v.available/2**30,'system_used_gib':v.used/2**30})
        except psutil.Error:pass
    def loop(self):
        while not self.done.is_set():self.sample();self.done.wait(.2)
    def stop(self):self.sample();self.done.set();self.t.join(2)
    def summary(self,phase=None):
        s=[x for x in self.samples if phase is None or x['phase']==phase]
        if not s:return {}
        return {'peak_working_set_mib':max(x['working_set_mib'] for x in s),'peak_private_mib':max(x['private_mib'] for x in s),'os_peak_working_set_mib':max(x['os_peak_working_set_mib'] for x in s),'os_peak_pagefile_mib':max(x['os_peak_pagefile_mib'] for x in s),'minimum_system_available_gib':min(x['system_available_gib'] for x in s),'peak_system_used_gib':max(x['system_used_gib'] for x in s),'memory_samples':len(s)}

def flags(text):
    lines=[l.strip() for l in text.splitlines() if l.strip()]
    zh=len(re.findall(r'[\u4e00-\u9fff]',text));letters=len(re.findall(r'[A-Za-z\u4e00-\u9fff]',text))
    return {**base.line_repeat_flags(text),'characters':len(text),'chinese_characters':zh,'chinese_ratio':zh/max(1,letters),'predominantly_chinese':zh>=20 and zh/max(1,letters)>.5,'normal_title_body':len(lines)>=2 and len(lines[0])<=120 and len(''.join(lines[1:]))>=40,'empty':not text.strip(),'garbled':'\ufffd' in text or any(ord(c)<9 for c in text),'non_task':zh<20 or len(lines)<2,'continuous_repeat_ge3':any(re.search(rf'(.{{{n}}})\1\1',re.sub(r'\s+','',text)) for n in range(8,min(81,len(text)//3+1)))}

def stream(messages,key,budget,path):
    payload={'messages':messages,'max_tokens':budget,'stream':True,'stream_options':{'include_usage':True},'cache_prompt':False}
    if key.startswith('lfm'):payload.update(LFM_SAMPLING)
    else:payload.update({'temperature':0,'enable_thinking':False})
    atomic(path.with_suffix('.request.json'),payload)
    req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    started=time.perf_counter();first=None;pieces=[];usage={};timings={};finish=None
    with path.with_suffix('.sse.jsonl').open('w',encoding='utf-8') as log:
        try:
            with urllib.request.urlopen(req,timeout=1800) as r:
                for line in r:
                    s=line.decode('utf-8').strip()
                    if not s.startswith('data:'):continue
                    s=s[5:].strip()
                    if s=='[DONE]':continue
                    obj=json.loads(s); log.write(json.dumps(obj,ensure_ascii=False)+'\n');log.flush()
                    usage=obj.get('usage') or usage;timings=obj.get('timings') or timings
                    for ch in obj.get('choices',[]):
                        finish=ch.get('finish_reason') or finish
                        content=ch.get('delta',{}).get('content')
                        if content:
                            if first is None:first=time.perf_counter()-started
                            pieces.append(content)
        except Exception as e:
            atomic(path.with_suffix('.failure.json'),{'error_type':type(e).__name__,'partial_output':''.join(pieces),'elapsed':time.perf_counter()-started})
            raise
    elapsed=time.perf_counter()-started;text=''.join(pieces)
    gen=timings.get('predicted_ms',elapsed*1000)/1000
    return {'output':text,'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':usage.get('prompt_tokens',timings.get('prompt_n')),'generated_tokens':usage.get('completion_tokens',timings.get('predicted_n')),'first_token_seconds':first,'total_seconds':elapsed,'generation_seconds':gen,'tokens_per_second':timings.get('predicted_per_second'),'characters_per_second':len(text)/elapsed,'decode_characters_per_second':len(text)/gen,'finish_reason':finish,'eos':finish=='stop','max_token_stop':finish=='length','usage':usage,'timings':timings,**flags(text)}

def smoke_inputs():
    from demo.schemas import WritingRequest
    from demo.agents import RepairedLoRAWriter
    req=WritingRequest('湖北移动完成武汉某产业园区5G网络优化','公司要闻','2026年8月28日，湖北移动完成武汉某产业园区5G网络优化工作。本次优化覆盖园区办公区、生产区和公共区域，共完成12处网络点位调整。优化完成后，园区重点区域5G网络覆盖得到改善。此次工作由湖北移动网络技术团队实施，后续将根据园区实际使用情况持续开展网络质量监测。','标题简洁明确；首段交代时间、主体和事件；第二段说明优化范围和具体工作；末段说明后续安排。不要添加材料中未提供的行业地位、经济价值或社会效益。')
    # Task content retained; avoid assigning a Qwen-specific persona to LFM.
    m=RepairedLoRAWriter._messages(req);m[0]['content']=m[0]['content'].replace('你是 Repaired LoRA Writer。','')
    return [('english_sanity',[{'role':'user','content':'What is the capital of France? Answer in one short sentence.'}],128),('chinese_sanity',[{'role':'user','content':'请用中文简短说明5G网络是什么，不超过两句话。'}],256),('current_smoke_5g',m,2048)]+[(r['sample_id'],r['messages'][:-1],2048) for r in inputs()[:2]]

def run(key,phase):
    dest=INTERIM/phase/key;dest.mkdir(parents=True,exist_ok=True)
    candidates=smoke_inputs() if phase=='smoke' else [(r['sample_id'],r['messages'][:-1],2048) for r in inputs()]
    if all((dest/(sid+'.json')).exists() for sid,_,_ in candidates):return
    assert not [p for p in psutil.process_iter(['name']) if 'llama-server' in (p.info['name'] or '').lower()], 'another model server is active'
    model=MODELS[key];integrity=read(ROOT/'model_integrity.json');assert any(v['path']==str(model) and v['passed'] for v in integrity.values())
    session_id=time.strftime('%Y%m%d_%H%M%S');log=ROOT/'logs'/f'{phase}_{key}_{session_id}.stderr.log'
    args=[str(SERVER),'-m',str(model),'-c','4096','-t','16','-tb','16','-ngl','0','--device','none','--no-op-offload','--no-kv-offload','--fit','off','-np','1','--host','127.0.0.1','--port',str(PORT),'--offline','--cache-ram','0','--no-cache-prompt','-v']
    if key.startswith('lfm'):args+=['--chat-template-file',str(INTERIM/'lfm_f16_native_template.jinja')]
    if key=='qwen_q4_lora':args+=['--lora',str(ADAPTER)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    for k in list(env):
        if k.startswith('LLAMA_ARG_'):env.pop(k)
    before=gpu();started=time.perf_counter();p=None;mon=None
    session={'condition':key,'phase':phase,'args':args,'gpu_before':before,'log':str(log),'model_sha256':next(v['sha256'] for v in integrity.values() if v['path']==str(model)),'system_ram_bytes':psutil.virtual_memory().total}
    session_path=dest/f'session_{session_id}.json'
    with log.open('wb') as err, log.with_suffix('.stdout.log').open('wb') as out:
        try:
            p=subprocess.Popen(args,cwd=SERVER.parent,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            mon=Monitor(p)
            while time.perf_counter()-started<240:
                if p.poll() is not None:raise RuntimeError('server load failed; inspect saved stderr')
                try:
                    if get('/health').get('status')=='ok':break
                except OSError:time.sleep(.2)
            else:raise TimeoutError('load timeout')
            session['load_seconds']=time.perf_counter()-started;session['gpu_loaded']=gpu();session['load_memory']=mon.summary('load')
            props=get('/props');atomic(dest/'server_props.json',props)
            rendered=[]
            for sid,messages,_ in candidates:
                req=urllib.request.Request(f'http://127.0.0.1:{PORT}/apply-template',data=json.dumps({'messages':messages,'add_generation_prompt':True},ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=10) as rr:rendered.append({'sample_id':sid,**json.load(rr)})
            jsonl(dest/'rendered_prompts.jsonl',rendered)
            if key=='qwen_q4_lora':
                session['lora_adapters']=get('/lora-adapters');assert session['lora_adapters'][0]['scale']==1.0
            mon.phase='warmup';atomic(dest/f'warmup_{session_id}.json',stream([{'role':'user','content':'你好，请简短回答。'}],key,64,dest/f'warmup_{session_id}'))
            for sid,messages,budget in candidates:
                target=dest/(sid+'.json')
                if target.exists():continue
                mon.phase=sid;mon.sample()
                response=stream(messages,key,budget,target);mon.sample()
                response.update({'condition':key,'sample_id':sid,'phase':phase,'prompt_sha256':digest(messages),'session_id':session_id,'resource_metrics':mon.summary(sid),'gpu_after_sample':gpu()})
                if sid=='current_smoke_5g':response['anchor_check']={a:a in response['output'] for a in ['2026年8月28日','湖北移动','武汉某产业园区','12处','网络质量监测']}
                atomic(target,response)
                print(json.dumps({'phase':phase,'condition':key,'sample':sid,'tokens':response['generated_tokens'],'seconds':round(response['total_seconds'],2),'finish':response['finish_reason']},ensure_ascii=False),flush=True)
            session['status']='completed'
        except Exception as e:
            session['status']='failed';session['error_type']=type(e).__name__;raise
        finally:
            if mon:mon.stop();session['memory']=mon.summary();jsonl(dest/f'resources_{session_id}.jsonl',mon.samples)
            if p and p.poll() is None:p.terminate();p.wait(timeout=30)
            session['gpu_after_unload']=gpu();atomic(session_path,session)

def prepare():
    selected=inputs();integrity=read(ROOT/'model_integrity.json')
    for key,path,expected in [('qwen_q4_lora',MODELS['qwen_q4_lora'],'2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e'),('repaired_adapter',ADAPTER,'ac138773efced59b32b86f4ce18d6197f7af8d0d95fdeb8a4566cf2f145c6b1b')]:
        h=sha(path);assert h==expected
        integrity[key]={'path':str(path),'size_bytes':path.stat().st_size,'sha256':h,'expected_sha256':expected,'passed':True,'reused_read_only':True}
    atomic(ROOT/'model_integrity.json',integrity)
    files=[PROJECT/'PROJECT_SPEC.md',PROJECT/'PLAN.md',PROJECT/'DECISIONS.md',PROJECT/'demo/agents.py',PROJECT/'demo/api_client.py',PROJECT/'demo/schemas.py',PROJECT/'demo/scoring.py',PROJECT/'src/training/lora_v2.py',base.PACK,base.VALIDATION,SERVER]
    if not (INTERIM/'protected_hashes.json').exists():atomic(INTERIM/'protected_hashes.json',{str(p):sha(p) for p in files})
    export('experiment_manifest.json',{'status':'RUNNING','root':str(ROOT),'source_project_read_only':True,'final_test_accessed':False,'validation':read(INTERIM/'validation_provenance.json'),'conditions':CONDITIONS,'runtime':{'server':str(SERVER),'sha256':sha(SERVER),'version':'b10822 c457e3bf7','threads':16,'threads_batch':16,'context':4096,'max_tokens':2048,'gpu_layers':0,'devices':'none','op_offload':False,'kv_offload':False,'parallel_models':False,'cache_prompt':False,'cache_ram_mib':0,'warmup_per_session':1},'lfm_sampling':LFM_SAMPLING,'qwen_sampling':{'temperature':0,'enable_thinking':False,'lora_scale':1.0},'template':'embedded model-native chat template; captured in server_props.json','protocol':'single-candidate blind; one batch interleaved across condition and sample','measurement_notes':['process start-to-health includes llama empty warmup and OS page-cache effects; not disk cold-load benchmark','client TTFT is first nonempty content; latency covers request to stream completion','characters/s uses Python Unicode length / end-to-end latency','memory sampled every 0.2s; OS peak working set/pagefile also retained','single run/seed; no thermal or affinity control']})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','smoke','validation']);ap.add_argument('--condition',choices=CONDITIONS);a=ap.parse_args()
    if a.phase=='prepare':prepare()
    else:
        for key in ([a.condition] if a.condition else CONDITIONS[:3] if a.phase=='smoke' else CONDITIONS):run(key,a.phase)

from common import *
import argparse
import dataclasses
import random
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from openai import OpenAI
from demo.config import APISettings
from demo.agents import ReviewerAgent, ReviserAgent, EditorialJudgeAgent
from demo.api_client import OpenAICompatibleClient
from demo.schemas import WritingRequest, Draft, parse_review, parse_draft, parse_judge
from demo.lora_parser import parse_lora_writer_output
from demo.scoring import release_adjusted_score
from run_generation import flags

FROZEN=module('frozen_agents_runner',r'experiments/lightweight/private_workspace\closed_loop_2x2_20260907\run_round1.py')
EXPECTED=Path(r'experiments/lightweight/private_workspace\closed_loop_2x2_20260907\results\experiment_manifest.json')

def judge_messages(row,text):
    class Capture:
        def request_json(self,messages,parser,schema_name):self.messages=messages
    c=Capture();EditorialJudgeAgent(c).run(WritingRequest('capture','','capture'),Draft('capture','capture'))
    # Preserve the frozen six-dimensional system rubric, send one exact raw anonymous text.
    return [c.messages[0],{'role':'user','content':json.dumps({'confirmed_facts_and_original_task':row['messages'][:-1],'anonymous_candidate':text},ensure_ascii=False,indent=2)}]

def leak_gate(messages,row=None,text=None):
    body=json.loads(messages[1]['content'])
    if 'anonymous_candidate' in body:
        assert set(body)=={'confirmed_facts_and_original_task','anonymous_candidate'}
        assert isinstance(body['anonymous_candidate'],str)
        assert body['anonymous_candidate']==text
        assert body['confirmed_facts_and_original_task']==row['messages'][:-1]
        assert row['messages'][-1] not in body['confirmed_facts_and_original_task']
    assert not re.search(r'\b(?:LFM2?\.?5?|QAD|PTQ|F16|Q4_0|Q4_K_M|Repaired LoRA)\b',messages[1]['content'],re.I)

class Trace:
    def __init__(self,path):self.path=path;self.attempts=rows(path)
    def __call__(self,url,headers,body,timeout):
        payload=json.loads(body);messages=payload['messages']
        assert not re.search(r'\b(?:LFM2?\.?5?|QAD|PTQ|F16|Q4_0|Q4_K_M|Repaired LoRA)\b',messages[1]['content'],re.I)
        base_url=url.rsplit('/chat/completions',1)[0]
        key=headers['Authorization'].removeprefix('Bearer ')
        for attempt in range(5):
            started=time.perf_counter()
            event={'request':payload,'request_hash':digest(payload),'started_epoch':time.time()}
            try:
                # No hidden SDK retries: every network attempt is accounted for below.
                with OpenAI(api_key=key,base_url=base_url,timeout=240,max_retries=0) as client:
                    response=client.chat.completions.create(**payload).model_dump()
                event.update({'success':True,'latency_seconds':time.perf_counter()-started,'response':FROZEN.sanitize_response(response)})
                self.attempts.append(event);jsonl(self.path,self.attempts)
                return response
            except Exception as e:
                # Error text may contain transport metadata; keep only type and numeric HTTP status.
                status=getattr(e,'status_code',None)
                event.update({'success':False,'latency_seconds':time.perf_counter()-started,'error_type':type(e).__name__,'http_status':status})
                self.attempts.append(event);jsonl(self.path,self.attempts)
                if status in (401,403) or attempt==4:raise RuntimeError('API request failed; safe attempt trace saved') from None
                time.sleep(min(30,2**attempt*2))

def stage(settings,stage,sid,condition,row,text,review=None):
    directory=INTERIM/'api'/stage;directory.mkdir(parents=True,exist_ok=True)
    path=directory/(digest([sid,condition,stage])[:24]+'.json')
    request=FROZEN.writing_request(row)
    messages=judge_messages(row,text) if stage=='judge' else None
    if stage=='judge':leak_gate(messages,row,text)
    h=digest({'stage':stage,'row':row['messages'][:-1],'text':text,'review':review.as_dict() if review else None,'model':settings.model,'protocol':'single-candidate-frozen-six-dimension-v1','prompt_contract':messages if stage=='judge' else FROZEN.prompt_captures()})
    if path.exists():
        old=read(path)
        assert old['input_hash']==h
        if old['status']=='completed':return old
        atomic(path.with_suffix('.failed_'+str(time.time_ns())+'.json'),old)
    trace=Trace(path.with_suffix('.attempts.jsonl'))
    client=OpenAICompatibleClient(settings,transport=trace)
    started=time.perf_counter()
    try:
        if stage=='judge':
            parsed=client.request_json(messages,parse_judge,'Editorial Judge')
            result=dataclasses.asdict(parsed);result['raw_editorial']=parsed.raw_total
            result['release_adjusted']=release_adjusted_score(parsed.raw_total,len(parsed.major_release_risks),parsed.publishable,len(parsed.unsupported_claims))
        else:
            draft=parse_lora_writer_output(text,{})
            parsed=ReviewerAgent(client).run(request,draft) if stage=='reviewer' else ReviserAgent(client).run(request,draft,review)
            result=parsed.as_dict()
        record={'status':'completed','stage':stage,'sample_id':sid,'condition_local_only':condition,'input_hash':h,'result':result,'latency_seconds':time.perf_counter()-started,'api_calls':len(trace.attempts),'usage':FROZEN.sum_usage(trace.attempts),'trace':str(trace.path)}
    except Exception as e:
        record={'status':'failed','stage':stage,'sample_id':sid,'condition_local_only':condition,'input_hash':h,'error_type':type(e).__name__,'api_calls':len(trace.attempts),'usage':FROZEN.sum_usage(trace.attempts),'trace':str(trace.path)}
    atomic(path,record)
    return record

def generation():
    selected=inputs()
    return {k:{r['sample_id']:read(INTERIM/'validation'/k/(r['sample_id']+'.json')) for r in selected} for k in CONDITIONS}

def closed_loop(settings,g):
    qad=list(g['lfm_qad'].values());n=len(qad)
    counts={k:sum(bool(x[k]) for x in qad) for k in ['predominantly_chinese','normal_title_body','severe_ge_5','empty','garbled','non_task','continuous_repeat_ge3']}
    bad=sum(x['empty'] or x['garbled'] or x['non_task'] for x in qad)
    passed=counts['predominantly_chinese']>n/2 and counts['normal_title_body']>n/2 and counts['severe_ge_5']/n<.2 and bad/n<.2
    export('qad_draft_gate.json',{'passed':passed,'n':n,'counts':counts,'bad_samples':bad,'predeclared_rule':'Chinese >50%, title/body >50%, severe line repetition <20%, empty/garbled/non-task <20%','does_not_mean_publishable':True})
    if not passed:return []
    assert FROZEN.prompt_captures()==read(EXPECTED)['prompt_and_code_hashes']
    closed=[]
    for row in inputs():
        sid=row['sample_id'];text=g['lfm_qad'][sid]['output'];r=stage(settings,'reviewer',sid,'lfm_qad',row,text)
        if r['status']!='completed':raise RuntimeError('Reviewer pending; resume supported')
        review=parse_review(r['result']);reviser=None
        if review.passed:final=text
        else:
            reviser=stage(settings,'reviser',sid,'lfm_qad',row,text,review)
            if reviser['status']!='completed':raise RuntimeError('Reviser pending; resume supported')
            d=parse_draft(reviser['result']);final=d.title+'\n\n'+d.body
        item={'sample_id':sid,'draft':text,'final':final,'draft_sha256':hashlib.sha256(text.encode()).hexdigest(),'final_sha256':hashlib.sha256(final.encode()).hexdigest(),'reviewer_decision':'PASS' if review.passed else 'FAIL','issues':[x.as_dict() for x in review.issues],'revision_triggered':not review.passed,'reviewer':r,'reviser':reviser,'final_stability':flags(final),'char_similarity_ratio':SequenceMatcher(None,text,final,autojunk=False).ratio()}
        atomic(INTERIM/'closed_loop'/(sid+'.json'),item);closed.append(item)
        export_rows('closed_loop_results.jsonl',closed)
        print(json.dumps({'stage':'closed_loop','sample':sid,'review':item['reviewer_decision'],'issues':len(item['issues']),'complete':len(closed)}),flush=True)
    return closed

def judge_batch(settings,g,closed):
    by_id={r['sample_id']:r for r in inputs()};conditions=CONDITIONS+(['lfm_qad_final'] if closed else [])
    texts={(k,sid):r['output'] for k,rr in g.items() for sid,r in rr.items()}
    texts.update({('lfm_qad_final',r['sample_id']):r['final'] for r in closed})
    order=list(texts);random.Random(42).shuffle(order)
    for i in range(1,len(order)):
        if order[i][0]==order[i-1][0]:
            j=next((j for j in range(i+1,len(order)) if order[j][0]!=order[i-1][0]),None)
            if j is not None:order[i],order[j]=order[j],order[i]
    export('editorial_batch_manifest.json',{'conditions_local_only':conditions,'n_requests_planned':len(order),'sample_atomic_resume':True,'order_local_only':[{'condition':k,'sample_id':sid} for k,sid in order],'single_candidate':True,'reference_provided':False,'old_scores_provided':False,'candidate_identity_provided':False,'judge_model':settings.model,'temperature':0,'schema_retry':1,'network_retry_max':4,'draft_score_reused_in_closed_loop_comparison':True,'pass_final_independently_judged':True})
    scores=[]
    for k,sid in order:
        r=stage(settings,'judge',sid,k,by_id[sid],texts[(k,sid)])
        scores.append(r);export_rows('editorial_results.jsonl',scores)
        print(json.dumps({'stage':'judge','complete':sum(x['status']=='completed' for x in scores),'planned':len(order),'status':r['status']}),flush=True)
    if any(r['status']!='completed' for r in scores):raise RuntimeError('Incomplete Judge batch; resume pending samples')

if __name__=='__main__':
    s=APISettings.from_env();assert s.is_configured and s.model==FROZEN.MODEL_REQUIRED
    g=generation();closed=closed_loop(s,g);judge_batch(s,g,closed)

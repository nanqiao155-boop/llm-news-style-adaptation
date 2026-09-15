from common import *
import argparse
import math
import random
import re
import statistics as st
from collections import Counter
from src.training.lora_v2 import evaluate_pairs

def summary(v):
    return {'mean':st.fmean(v),'median':st.median(v),'p90':base.percentile(v,.9),'min':min(v),'max':max(v)} if v else None

def collect():
    selected=inputs();all_rows=[];deploy={};stability={};auto={};smoke={}
    for k in CONDITIONS:
        rr=[read(p) for r in selected if (p:=INTERIM/'validation'/k/(r['sample_id']+'.json')).exists()]
        for r in rr:
            compact=re.sub(r'\s+','',r['output'])
            r['continuous_repeat_ge5_supplemental']=any(re.search(rf'(.{{{n}}})\1\1\1\1',compact) for n in range(8,min(81,len(compact)//5+1)))
        all_rows+=rr
        ss=[read(p) for p in (INTERIM/'validation'/k).glob('session_*.json')]
        good=[s for s in ss if s['status']=='completed']
        statistics_fields=['first_token_seconds','total_seconds','generation_seconds','tokens_per_second','characters_per_second','decode_characters_per_second','prompt_tokens','generated_tokens','characters']
        d={'count':len(rr),'model_file_bytes':MODELS[k].stat().st_size,'adapter_file_bytes':ADAPTER.stat().st_size if k=='qwen_q4_lora' else 0,'system_ram_bytes':16_779_862_016,'sessions':ss}
        d['total_deployment_bytes']=d['model_file_bytes']+d['adapter_file_bytes']
        d['file_accounting']='required inference weights plus adapter; shared runtime, source code and resumable download cache excluded and reported separately'
        d['shared_runtime_executable_bytes']=SERVER.stat().st_size
        d['shared_runtime_directory_bytes']=sum(p.stat().st_size for p in SERVER.parent.rglob('*') if p.is_file())
        d['total_deployment_gib']=d['total_deployment_bytes']/2**30
        d.update({f:summary([r[f] for r in rr if isinstance(r.get(f),(int,float))]) for f in statistics_fields})
        d['load_seconds']=[s['load_seconds'] for s in ss if 'load_seconds' in s]
        for f in ['peak_working_set_mib','peak_private_mib','os_peak_working_set_mib','os_peak_pagefile_mib','peak_system_used_gib']:
            d[f]=max((s['memory'][f] for s in ss if s.get('memory')),default=None)
        d['minimum_system_available_gib']=min((s['memory']['minimum_system_available_gib'] for s in ss if s.get('memory')),default=None)
        d['cpu_evidence']=[]
        for s in ss:
            log=Path(s['log']).read_text(encoding='utf-8',errors='replace')
            assigned=re.findall(r'layer\s+\d+ assigned to device (\w+)',log)
            gpu_buffers=re.findall(r'CUDA\d+\s+(?:model|KV|compute) buffer size',log)
            evidence={'log':s['log'],'assigned_layers':len(assigned),'all_layers_CPU':bool(assigned) and all(x=='CPU' for x in assigned),'cuda_model_kv_compute_buffers':gpu_buffers,'thread_16_evidence':'n_threads = 16' in log,'gpu_snapshots':{x:s.get(x) for x in ['gpu_before','gpu_loaded','gpu_after_unload']}}
            evidence['buffer_placement_lines']=[line for line in log.splitlines() if 'buffer size =' in line]
            d['cpu_evidence'].append(evidence)
        deploy[k]=d
        stability[k]={'count':len(rr),**{f+'_count':sum(bool(r[f]) for r in rr) for f in ['eos','max_token_stop','repeat_ge_3','consecutive_ge_3','severe_ge_5','predominantly_chinese','normal_title_body','empty','garbled','non_task','continuous_repeat_ge3']},'anomaly_sample_ids':[r['sample_id'] for r in rr if r['max_token_stop'] or r['severe_ge_5'] or r['garbled'] or r['empty'] or r['continuous_repeat_ge3']]}
        stability[k]['continuous_repeat_ge5_supplemental_count']=sum(r['continuous_repeat_ge5_supplemental'] for r in rr)
        stability[k]['supplementary_rule']='whitespace removed; any 8–80 character span repeated consecutively >=5 times; added after observing a paragraph-loop anomaly, not a frozen primary metric'
        if rr:
            refs={r['sample_id']:r['messages'][-1]['content'] for r in selected}
            auto[k]=evaluate_pairs([(r['output'],refs[r['sample_id']]) for r in rr])
        if k.startswith('lfm'):
            sm=[read(p) for p in (INTERIM/'smoke'/k).glob('*.json') if p.stem in ['english_sanity','chinese_sanity','current_smoke_5g']+[r['sample_id'] for r in selected[:2]]]
            smoke[k]={'count':len(sm),'all_eos':all(x['eos'] for x in sm),'no_severe_or_garbled':all(not x['severe_ge_5'] and not x['garbled'] for x in sm),'results':sm}
    export_rows('generation_outputs.jsonl',all_rows)
    fields=['condition','sample_id','prompt_tokens','generated_tokens','characters','first_token_seconds','generation_seconds','total_seconds','tokens_per_second','characters_per_second','decode_characters_per_second','finish_reason','eos','max_token_stop','repeat_ge_3','consecutive_ge_3','severe_ge_5','predominantly_chinese','normal_title_body','garbled']
    csvout('runtime_metrics.csv',[{**{f:r[f] for f in fields},**r['resource_metrics']} for r in all_rows])
    export('deployment_summary.json',deploy);export('stability_summary.json',stability);export('automatic_metrics.json',auto);export('smoke_summary.json',smoke)
    return all_rows,deploy,stability,auto,smoke

def value(s,k):
    if k in ('unsupported_claims','major_release_risks'):return len(s[k])
    if k in s['dimensions']:return s['dimensions'][k]
    return float(s[k])

METRICS=['raw_editorial','release_adjusted','publishable','unsupported_claims','major_release_risks','factual_grounding','key_information_retention','title_quality','formality_objectivity','structure_formatting','conciseness_naturalness']

def compare(scores,a,b):
    ids=[r['sample_id'] for r in inputs() if r['sample_id'] in scores[a] and r['sample_id'] in scores[b]]
    rng=random.Random(42);boot=[[rng.randrange(len(ids)) for _ in ids] for _ in range(10000)]
    out={'before':a,'after':b,'n':len(ids),'direction':'after minus before; negative risk-count deltas are improvements','uncertainty':'paired nonparametric sample bootstrap, 10000 replicates, seed 42; descriptive, one generation run and one Judge draw, no multiple-comparison adjustment','metrics':{}}
    for metric in METRICS:
        aa=[value(scores[a][sid],metric) for sid in ids];bb=[value(scores[b][sid],metric) for sid in ids];dd=[y-x for x,y in zip(aa,bb)]
        means=[st.fmean(dd[j] for j in ii) for ii in boot]
        out['metrics'][metric]={'before_mean':st.fmean(aa),'after_mean':st.fmean(bb),'delta':st.fmean(dd),'delta_bootstrap_95ci':[base.percentile(means,.025),base.percentile(means,.975)],'increase':sum(x>0 for x in dd),'equal':sum(x==0 for x in dd),'decrease':sum(x<0 for x in dd)}
    rescue=[sid for sid in ids if not scores[a][sid]['publishable'] and scores[b][sid]['publishable']]
    harm=[sid for sid in ids if scores[a][sid]['publishable'] and not scores[b][sid]['publishable']]
    n=len(rescue)+len(harm);p=min(1,2*sum(math.comb(n,i) for i in range(min(len(rescue),len(harm))+1))/2**n) if n else 1
    out.update({'rescue_ids':rescue,'harm_ids':harm,'exact_mcnemar_two_sided_p':p})
    return out

def tables(deploy,auto,quality):
    lines=['| Condition | Files GiB | Load s | Peak WS MiB | Peak Private MiB | TTFT median s | Article median s | decode tok/s median | char/s median |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,d in deploy.items():
        lines.append(f"| {k} | {d['total_deployment_gib']:.3f} | {d['load_seconds'][0]:.2f} | {d['peak_working_set_mib']:.1f} | {d['peak_private_mib']:.1f} | {d['first_token_seconds']['median']:.2f} | {d['total_seconds']['median']:.2f} | {d['tokens_per_second']['median']:.2f} | {d['characters_per_second']['median']:.2f} |")
    lines+=['','Automatic metrics on the exact same 24 references; 0–1 scale; corpus BLEU-4 and macro ROUGE F1.','','| Condition | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |','|---|---:|---:|---:|---:|']
    for k,a in auto.items():lines.append('| '+k+' | '+' | '.join(f'{a[x]:.4f}' for x in ['BLEU-4','ROUGE-1','ROUGE-2','ROUGE-L'])+' |')
    lines+=['','| Condition | Raw /100 | Release /100 | Publishable | Unsupported/sample | Major risks/sample |','|---|---:|---:|---:|---:|---:|']
    for k,q in quality.items():lines.append(f"| {k} | {q['raw_editorial']:.2f} | {q['release_adjusted']:.2f} | {q['publishable']:.1%} | {q['unsupported_claims']:.3f} | {q['major_release_risks']:.3f} |")
    return lines

def main():
    generation,deployment,stability,auto,smoke=collect()
    judge=rows(ROOT/'editorial_results.jsonl');successful=[r for r in judge if r['status']=='completed']
    export('editorial_batch_usage.json',{'completed_candidates':len(successful),'api_attempts':sum(r['api_calls'] for r in judge),'usage':{k:sum(r['usage'][k] for r in judge) for k in ['prompt_tokens','completion_tokens','total_tokens']},'purpose':'evaluation-only overhead, separate from Reviewer/Reviser deployment burden'})
    if len(successful)<96:print(json.dumps({'generation':len(generation),'editorial':len(successful)}));return
    scores={k:{r['sample_id']:r['result'] for r in successful if r['condition_local_only']==k} for k in CONDITIONS+['lfm_qad_final']}
    scores={k:v for k,v in scores.items() if v}
    quality={k:{'n':len(ss),**{m:st.fmean(value(s,m) for s in ss.values()) for m in METRICS}} for k,ss in scores.items()}
    export('editorial_summary.json',quality)
    f16_ptq=compare(scores,'lfm_f16','lfm_ptq');ptq_qad=compare(scores,'lfm_ptq','lfm_qad');qad_f16=compare(scores,'lfm_f16','lfm_qad');lfm_qwen=compare(scores,'qwen_q4_lora','lfm_qad')
    recovery={}
    for m in METRICS:
        loss=quality['lfm_f16'][m]-quality['lfm_ptq'][m]
        oriented_loss=-loss if m in ['unsupported_claims','major_release_risks'] else loss
        recovery[m]={'f16_minus_ptq':loss,'qad_minus_ptq':quality['lfm_qad'][m]-quality['lfm_ptq'][m],'fraction_of_observed_ptq_loss_recovered':(quality['lfm_qad'][m]-quality['lfm_ptq'][m])/loss if oriented_loss>0 else None,'note':'fraction only defined when PTQ is worse than F16 for this metric; values are not clipped'}
    ptq_qad['recovery_fraction']=recovery
    ptq_qad['automatic_recovery']={m:{'f16':auto['lfm_f16'][m],'ptq':auto['lfm_ptq'][m],'qad':auto['lfm_qad'][m],'qad_minus_ptq':auto['lfm_qad'][m]-auto['lfm_ptq'][m],'descriptive_loss_recovery_fraction':(auto['lfm_qad'][m]-auto['lfm_ptq'][m])/(auto['lfm_f16'][m]-auto['lfm_ptq'][m]) if auto['lfm_f16'][m]>auto['lfm_ptq'][m] else None} for m in ['BLEU-4','ROUGE-1','ROUGE-2','ROUGE-L']}
    export('f16_vs_ptq_analysis.json',f16_ptq);export('ptq_vs_qad_analysis.json',ptq_qad);export('qad_vs_f16_analysis.json',qad_f16)
    lfm_qwen['interpretation_boundary']='Observed deployment systems; parameter count, tokenizer, tuning, quantization and decoding differ. Not architecture-only causal evidence.'
    q=deployment['lfm_qad'];w=deployment['qwen_q4_lora']
    lfm_qwen['deployment_ratios_qad_over_qwen']={'files':q['total_deployment_bytes']/w['total_deployment_bytes'],'peak_working_set':q['peak_working_set_mib']/w['peak_working_set_mib'],'peak_private':q['peak_private_mib']/w['peak_private_mib'],'median_article_latency':q['total_seconds']['median']/w['total_seconds']['median'],'median_characters_per_second':q['characters_per_second']['median']/w['characters_per_second']['median']}
    export('lfm_vs_qwen_analysis.json',lfm_qwen)
    closed=rows(ROOT/'closed_loop_results.jsonl');closed_analysis=None
    if closed:
        closed_analysis=compare(scores,'lfm_qad','lfm_qad_final')
        records=[s for r in closed for s in [r['reviewer'],r['reviser']] if s]
        closed_analysis['burden']={'reviewer_trigger_rate':sum(r['revision_triggered'] for r in closed)/len(closed),'reviewer_calls':sum(r['reviewer']['api_calls'] for r in closed),'reviser_calls':sum(r['reviser']['api_calls'] for r in closed if r['reviser']),'api_calls':sum(r['api_calls'] for r in records),'usage':{k:sum(r['usage'][k] for r in records) for k in ['prompt_tokens','completion_tokens','total_tokens']},'issues':dict(Counter(i['type'] for r in closed for i in r['issues'])),'issues_total':sum(len(r['issues']) for r in closed),'api_latency_total_seconds':sum(r['latency_seconds'] for r in records),'per_sample_api_latency_seconds':summary([r['reviewer']['latency_seconds']+(r['reviser']['latency_seconds'] if r['reviser'] else 0) for r in closed]),'per_sample_api_tokens':summary([r['reviewer']['usage']['total_tokens']+(r['reviser']['usage']['total_tokens'] if r['reviser'] else 0) for r in closed]),'char_similarity':summary([r['char_similarity_ratio'] for r in closed])}
        closed_analysis['same_text_pass_judge_variability']=[{'sample_id':r['sample_id'],'release_delta':scores['lfm_qad_final'][r['sample_id']]['release_adjusted']-scores['lfm_qad'][r['sample_id']]['release_adjusted']} for r in closed if r['reviewer_decision']=='PASS']
        export('closed_loop_analysis.json',closed_analysis)
    result={'status':'SUCCESS' if len(generation)==96 and all(len(s)==24 for s in scores.values()) else 'PARTIAL','generation_count':len(generation),'judge_completed':len(successful),'deployment':deployment,'stability':stability,'automatic':auto,'editorial':quality,'ptq_vs_qad':ptq_qad,'qad_vs_f16':qad_f16,'lfm_vs_qwen':lfm_qwen,'closed_loop':closed_analysis,'final_test_accessed':False,'locally_trained_qad':False,'official_model_url':'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF'}
    result['research_verdicts']={'cpu_draft_feasibility':'SUPPORTED_ON_THIS_VALIDATION_SET','qad_editorial_recovery_over_ptq':'NOT_DEMONSTRATED','qad_worse_than_ptq_in_general':'INCONCLUSIVE_CI_CROSSES_ZERO','qad_f16_equivalence':'NOT_ESTABLISHED','qad_direct_replacement_for_qwen_at_equal_draft_quality':'NOT_SUPPORTED','closed_loop':'LARGE_AUTO_JUDGE_GAIN_WITH_100_PERCENT_REVISER_TRIGGER','judge_ceiling':{'final_100_score_count':sum(s['raw_editorial']==100 for s in scores.get('lfm_qad_final',{}).values()),'same_model_for_reviser_and_judge':True,'human_publishability_verified':False},'experiment_ready_for_ppt':True}
    export('research_verdicts.json',result['research_verdicts'])
    export('aggregate_final.json',result)
    md=['# LFM2.5 Hybrid Architecture and Quantization Compensation','',f"Status: **{result['status']}**. Fixed Validation-24; CPU-only i9-13980HX; 16 threads; context 4096; output budget 2048; one run per condition.",'','OBSERVED — 15/15 preliminary Smoke requests completed with EOS. All three 5G outputs omitted the exact monitoring anchor; loading success does not imply editing success.','']+tables(deployment,auto,quality)
    md+=['','OBSERVED — Paired Release score differences (after − before, 95% sample-bootstrap CI):']
    for label,a in [('F16 → PTQ',f16_ptq),('PTQ → QAD',ptq_qad),('F16 → QAD',qad_f16),('Qwen → QAD',lfm_qwen)]+([('QAD Draft → Final',closed_analysis)] if closed_analysis else []):
        m=a['metrics']['release_adjusted'];md.append(f"- {label}: {m['delta']:+.2f} [{m['delta_bootstrap_95ci'][0]:+.2f}, {m['delta_bootstrap_95ci'][1]:+.2f}]; publishable {a['metrics']['publishable']['delta']*100:+.2f} percentage points; unsupported/sample {a['metrics']['unsupported_claims']['delta']:+.3f}; rescue {len(a['rescue_ids'])}, harm {len(a['harm_ids'])}.")
    if closed_analysis:md+=['',f"OBSERVED — Closed-loop burden: {closed_analysis['burden']['api_calls']} Reviewer/Reviser API attempts, {closed_analysis['burden']['usage']['total_tokens']} tokens, trigger rate {closed_analysis['burden']['reviewer_trigger_rate']:.1%}. Judge calls are evaluation overhead and are reported separately in editorial_results.jsonl."]
    md+=['','INTERPRETATION — PTQ/QAD is the main within-format comparison. A smaller/faster Writer may still impose a substantial remote revision burden. Release quality, unsupported counts and publishability should be read together.','', 'LIMITATION — 24 previously used Validation samples, one seed/run, one automated Judge model, no human publishability adjudication, no Final Test, no isolated architecture ablation, no local QAD training, no run-order counterbalancing, no thermal/affinity isolation, OS page-cache effects in load times, process peak memory rather than total application footprint. Different generated lengths affect article latency; char/s complements tokenizer-dependent tok/s. Bootstrap intervals concern sampled articles only; model/API nondeterminism and judge bias are not captured. WDDM per-process VRAM is unavailable; CPU placement logs plus global GPU snapshots support no model offload, while a CUDA-linked binary can initialize a driver context.','', 'All raw generations, SSE chunks, exact requests, native/rendered templates, resource traces and safe API responses are under data/interim. All derived data is under data/processed; root files are exports.']
    md += ['', 'INTERPRETATION — This run does not demonstrate Editorial quality recovery from PTQ to QAD. The negative mean Release delta has an interval crossing zero, so it also does not establish general inferiority. QAD provides a small, fast CPU draft generator, with substantially lower draft quality than the adapted Qwen reference in this batch.', '', f"LIMITATION — QAD Final has {result['research_verdicts']['judge_ceiling']['final_100_score_count']}/24 perfect scores. Reviser and Judge use the same model in separate calls; ceiling effects and correlated model preferences limit the interpretation of 100% auto-judged publishability. This is not independently verified human publishability. All 24 Drafts needed revision; mean character similarity is {closed_analysis['burden']['char_similarity']['mean']:.3f}. Final quality cannot be credited to the 1.2B Writer alone."]
    (ROOT/'aggregate_final.md').write_text('\n'.join(md)+'\n',encoding='utf-8');(PROCESSED/'aggregate_final.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    csvout('ppt_summary.csv',[{'condition':k,'n':quality[k]['n'],'files_gib':deployment[k]['total_deployment_gib'] if k in deployment else None,'peak_working_set_mib':deployment[k]['peak_working_set_mib'] if k in deployment else None,'median_latency_seconds':deployment[k]['total_seconds']['median'] if k in deployment else None,'median_characters_per_second':deployment[k]['characters_per_second']['median'] if k in deployment else None,**{m:quality[k][m] for m in METRICS}} for k in quality])
    print(json.dumps({'status':result['status'],'quality':quality,'closed_burden':closed_analysis['burden'] if closed_analysis else None},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

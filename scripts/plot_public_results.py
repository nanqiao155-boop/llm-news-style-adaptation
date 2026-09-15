"""Redraw neutral figures from published aggregates only; no model or API execution."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'

def rows(path):
    with (ROOT/path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT/(name+'.png'), dpi=160, facecolor='white')
    plt.close(fig)

def bar(ax, names, values, label):
    ax.bar(names, values, color=['#52647a','#387c78','#a17f50'][:len(names)])
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)

def main():
    plt.rcParams.update({'font.size':11})
    fig, axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,label in zip(axes,['lr','rank','batch'],['learning_rate','rank','effective_batch']):
        data=rows('results/lora/'+key+'.csv')
        # Formal summaries use a consistent metric name; parameter columns vary.
        param=label if label in data[0] else next(iter(data[0]))
        bar(ax,[r[param] for r in data],[float(r['BLEU-4']) for r in data],'BLEU-4')
        ax.set_title({'lr':'Learning rate','rank':'LoRA rank','batch':'Effective batch'}[key])
    save(fig,'lora_controls')
    data=json.loads((ROOT/'results/stability/summary.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4));bar(ax,['Original','Repaired'],[data['original_severe'],data['repaired_severe']],'Severe repetition count')
    for x,y in enumerate([data['original_severe'],data['repaired_severe']]):ax.text(x,y+1,str(y)+'/234',ha='center')
    ax.set_ylim(0,65);ax.set_title('Full Validation: checkpoint selection + strength 0.50');save(fig,'stability')
    data=rows('results/final_system/final_test.csv');fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,k,title in zip(axes,['release_adjusted_score','publishable_rate','unsupported_claims_per_sample'],['Release-Adjusted','Publishable (%)','Unsupported/sample']):
        bar(ax,[r['model'] for r in data],[float(r[k]) for r in data],title)
        ax.set_title('Final Test (n=50)')
    save(fig,'final_test')
    data=rows('results/lightweight/qwen_ppt_same_source_system_tradeoff.csv');d={r['Metric']:r for r in data}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,k in zip(axes,['Peak VRAM (MiB)','Writer tok/s','Final Release-Adjusted']):
        bar(ax,['Q4','Q6','Q8'],[float(d[k][q]) for q in ['Q4','Q6','Q8']],k)
        ax.set_title('Same-source Qwen (n=24)')
    save(fig,'qwen_deployment')
    data=[r for r in rows('results/lightweight/lfm_ppt_summary.csv') if r['condition'] in ['lfm_f16','lfm_ptq','lfm_qad']]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,k,label in zip(axes,['peak_working_set_mib','median_latency_seconds','median_characters_per_second'],['Peak working set (MiB)','Median latency (s)','Median characters/s']):
        bar(ax,['F16','PTQ','QAD'],[float(r[k]) for r in data],label);ax.set_title('LFM CPU-only (n=24)')
    save(fig,'lfm_cpu')
    data=json.loads((ROOT/'results/lightweight/lfm_independent_audit.json').read_text(encoding='utf-8'))['stages']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    bar(axes[0],['Draft','Final'],[100*data[k]['publishable_rate'] for k in ['Draft','Final']],'Independent publishable (%)')
    bar(axes[1],['Draft','Final'],[data[k]['issue_means']['unsupported_claim_count'] for k in ['Draft','Final']],'Independent unsupported/sample')
    fig.suptitle('Independent QAD factual audit (n=24)');save(fig,'independent_audit')
    print('Generated six aggregate-only PNG figures')

if __name__=='__main__':main()

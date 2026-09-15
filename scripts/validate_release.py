"""Offline public-release checks. Reports never contain matched secret values."""
from pathlib import Path
import ast
import csv
import importlib
import json
import re
import sys
from urllib.parse import urlparse
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPORTS = {'docs/validation_report.json','docs/secret_scan_report.csv','large_files_report.csv'}
BANNED_SUFFIXES = {'.safetensors','.bin','.gguf','.pt','.pth','.ckpt','.zip','.tar','.gz','.docx','.pptx','.pdf','.pyc'}
TEXT_SUFFIXES = {'.py','.md','.json','.jsonl','.csv','.yaml','.yml','.txt','.example'}
CREDENTIAL = re.compile(r'(?i)(?<![\w-])sk-[A-Za-z0-9_-]{20,}|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}')
LEXEMES = re.compile(r'(?i)api[_ -]?key|bearer|authorization|access_token|password|cookie|credential|oss|bailian|endpoint')
ABS_PATH = re.compile(r'(?i)(?<![A-Za-z])[A-Z]:[\\/]{1,2}(?:Users|yidongshixi|cm_lightweight_lab|github_release)|/(?:home|root|mnt)/[A-Za-z]')
PII = re.compile(r'(?<![\w])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w])|(?<!\d)1[3-9]\d{9}(?!\d)')
PUBLIC_HOSTS = {'huggingface.co','github.com','json-schema.org','www.10086.cn','10086.cn','127.0.0.1','localhost','api.example.invalid'}

def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    errors=[]; findings=[]; large=[]; counts={'python_compile':0,'json':0,'jsonl':0,'csv':0,'yaml':0,'markdown_links':0,'imports':0}
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if rel in REPORTS:continue
        if p.suffix.lower() in BANNED_SUFFIXES:errors.append({'path':rel,'type':'forbidden_file'})
        if p.name in {'.env','.env.local'} or (p.name.startswith('.env.') and p.name!='.env.example'):
            errors.append({'path':rel,'type':'private_env'})
        if p.stat().st_size > 50*1024*1024:large.append({'path':rel,'bytes':p.stat().st_size,'risk':'large_file','action':'exclude before publication'})
        if p.suffix not in TEXT_SUFFIXES and p.name!='.gitignore':continue
        try:t=p.read_text(encoding='utf-8-sig')
        except UnicodeError:errors.append({'path':rel,'type':'encoding'});continue
        for kind,pattern in [('credential_shape',CREDENTIAL),('personal_absolute_path',ABS_PATH),('personal_identifier_candidate',PII)]:
            hits=[i+1 for i,line in enumerate(t.splitlines()) if pattern.search(line)]
            if hits:
                findings.append({'path':rel,'type':kind,'risk':'review_required','action':'inspect without disclosing value','lines':';'.join(map(str,hits))})
                errors.append({'path':rel,'type':kind})
        for url in re.findall(r'https?://[^\s\x22\x27<>\)]+',t):
            host=urlparse(url).hostname
            if host and host not in PUBLIC_HOSTS and '{' not in host:
                findings.append({'path':rel,'type':'endpoint_candidate','risk':'review_required','action':'inspect host; value omitted','lines':''})
                errors.append({'path':rel,'type':'endpoint_candidate'})
        if LEXEMES.search(t):
            findings.append({'path':rel,'type':'credential_related_code_or_documentation','risk':'informational','action':'identifiers, empty placeholders, validation patterns or runtime variables; no credential value detected','lines':''})
        try:
            if p.suffix=='.py':
                compile(t,rel,'exec');counts['python_compile']+=1
                tree=ast.parse(t)
                for n in ast.walk(tree):
                    if isinstance(n,(ast.Assign,ast.AnnAssign)):
                        value=n.value
                        names=[x.id for x in getattr(n,'targets',[]) if isinstance(x,ast.Name)]
                        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):names.append(n.target.id)
                        if any(re.fullmatch(r'(?i)(?:.*_)?(?:api_key|password|access_token|secret_key)',x) for x in names) and isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value and not value.value.startswith(('example','placeholder')):
                            errors.append({'path':rel,'type':'literal_credential_assignment'})
            elif p.suffix=='.json':json.loads(t);counts['json']+=1
            elif p.suffix=='.jsonl':
                rows=[json.loads(line) for line in t.splitlines() if line.strip()]
                if not rel.startswith('data/samples/') or any(r.get('synthetic') is not True for r in rows):errors.append({'path':rel,'type':'non_synthetic_jsonl'})
                counts['jsonl']+=1
            elif p.suffix=='.csv':
                parsed=list(csv.reader(t.splitlines()))
                if not parsed or any(len(row)!=len(parsed[0]) for row in parsed):raise ValueError('CSV width')
                counts['csv']+=1
            elif p.suffix in {'.yaml','.yml'}:
                import yaml
                yaml.safe_load(t);counts['yaml']+=1
            elif p.suffix=='.md':
                for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',t):
                    if target.startswith(('http:','https:','#','mailto:')):continue
                    target=target.split('#')[0]
                    counts['markdown_links']+=1
                    if not (p.parent/target).exists():errors.append({'path':rel,'type':'broken_link','target':target})
        except Exception as exc:errors.append({'path':rel,'type':'parse_or_compile','error_class':type(exc).__name__})
    # Import core reusable code only. Historical lightweight runners require excluded assets.
    for directory in ['src','demo','scripts']:
        for p in (ROOT/directory).rglob('*.py'):
            if p.name in {'app.py','plot_public_results.py','validate_release.py'}:continue
            name='.'.join(p.relative_to(ROOT).with_suffix('').parts)
            try:importlib.import_module(name);counts['imports']+=1
            except Exception as exc:errors.append({'path':p.relative_to(ROOT).as_posix(),'type':'import','error_class':type(exc).__name__})
    write_csv(ROOT/'large_files_report.csv',['path','bytes','risk','action'],large)
    write_csv(ROOT/'docs/secret_scan_report.csv',['path','type','risk','action','lines'],findings)
    secret_errors=[e for e in errors if e['type'] in {'credential_shape','literal_credential_assignment','private_env'}]
    result={'status':'PASS' if not errors and not large else 'REVIEW_REQUIRED','secret_status':'NO_CREDIBLE_SECRET' if not secret_errors else 'STOP_SECRET_FOUND','checks':counts,'large_files':len(large),'findings_without_values':errors,'scope':'All staging text, structured files, Python and file names; only newly generated PNG figures are binary. Original private files were excluded without reading credentials. No remote secret scanner or upload used.','limitations':['Regex and literal scans do not prove absence of every possible secret.','Historical lightweight reference imports and full model/API execution are excluded from runnable checks.']}
    (ROOT/'docs/validation_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())

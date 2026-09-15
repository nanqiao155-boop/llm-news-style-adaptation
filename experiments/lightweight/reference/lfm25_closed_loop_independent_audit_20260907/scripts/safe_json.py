"""Expose nested JSON structure without decoding excluded semantic values."""
import json
def value_end(raw,i):
    depth=0;quoted=False;escaped=False
    while i<len(raw):
        c=raw[i]
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':quoted=False
        elif c=='"':quoted=True
        elif c in '[{':depth+=1
        elif c in ']}':
            if depth==0:break
            depth-=1
        elif c==',' and depth==0:break
        i+=1
    return i
def spans(raw):
    raw=raw.lstrip('\ufeff \r\n\t');opening=raw[0]
    assert opening in '[{'
    out={} if opening=='{' else [];i=1;dec=json.JSONDecoder()
    while True:
        while raw[i].isspace() or raw[i]==',':i+=1
        if raw[i] in ']}':return out
        if opening=='{':
            key,i=dec.raw_decode(raw,i)
            while raw[i].isspace():i+=1
            assert raw[i]==':';i+=1
            while raw[i].isspace():i+=1
        end=value_end(raw,i);value=raw[i:end]
        if opening=='{':out[key]=value
        else:out.append(value)
        i=end
def leaf(raw,*keys):
    for k in keys:raw=spans(raw)[k]
    return json.loads(raw)

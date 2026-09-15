"""Versioned deterministic lexical evidence, not semantic factual judgments."""
import re, json, pathlib, unicodedata, collections, decimal
from prepare_audit import ROOT, dump, jsonl
VERSION='anchor-audit-1.0'
LEXICON={
'person':'段李娟 杨景康 杨杰 李跃 董昕 尚冰 王建宙 奚国华 何飚 胡翔 习近平',
'organization':'企业 企业通信集团公司 集团工会 全国总工会女职工委员会 中国国防邮电工会 中国电信 中国联通 工信部电信研究院 企业研究院 企业江苏公司 企业西藏公司 西藏移动 湖北移动 襄阳移动 洛阳移动 浙江移动 芯昇科技 Omdia GTI Orange 法国电信集团 台湾工业技术研究院 中华电信 远传电信 HTC MTK 鸿海 中国旅游集团 国务院国资委 咪咕公司 GSMA GSM协会 ITU 3GPP 解放军总医院 泰康同济医院 湖北省妇幼保健院光谷院区 火神山医院 雷神山医院 北京301医院 奥组委 香港大学 国家知识产权局',
'place':'北京 福建 榆中县 马坡乡 马莲滩村 福州 湖北 利川市 齐岳山 襄阳 河南 洛阳 浙江 台湾 新竹市 夏邑 黎平 宁夏 青岛 巴塞罗那 美国 欧洲 日本 韩国 俄罗斯 江苏 珠穆朗玛峰 珠峰 西藏 绒布寺 拉萨 武汉 青海 柴达木 海南省 白沙县 海南省白沙黎族自治县 岭尾村 荣邦乡 香港 深圳',
'product_business':'物流电子锁 宠物监控终端 车机 BASIC6 联创+计划 破风8676 算网大脑 企业01星 星核 5G视频客服 十百千万 梧桐大数据 梧桐大数据平台 和美乡途 九天AI 玩AI AI向导 AI旅拍 5G新通话 九天基础大模型V3.0 九天平台 九天 MoMA 灵犀2.0 灵犀智能体 AI云电脑 AI+新通话 宁夏水利大模型 人工智能安全评测平台 五岳纪元量子云平台 JOY电子商务网 高标党建 两个新型 AI+ 电子采购 企业采购 酒店预订 机票预订 采购信息发布 来电必复 10086 1008611 400-100-2-100 中国专利金奖 中国专利优秀奖 中国专利银奖 全国五一劳动奖章 全国五一巾帼奖章 全国五一巾帼标兵 央企楷模',
}
EXTRA_PRODUCTS=['MAT LOCK','TD-LTE Advanced','LTE TDD/FDD','Band 3','Band 41','Web 3.0','S+C+L','4G-LTE','TD-LTE','FDD-LTE','5G-A','Wi-Fi','GPS','IP67','VoLTE','RCS','WiMAX','NFC','LTE','PON','GSM','WLAN','TD','全IP','VR/AR','5G+AICDE','4G','5G','6G','3G','3D','B2B','4G移动互联网时代的创新与变革','下一代融合通信白皮书','接入网关的选择方法、系统及网关选择执行节点','信号传输系统以及相关装置']
for k in LEXICON: LEXICON[k]=LEXICON[k].split()
LEXICON['product_business']+=EXTRA_PRODUCTS
def norm(s): return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).replace('−','-').replace('％','%')
def han_number(s):
    digits={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    if s in digits:return str(digits[s])
    if all(c in digits for c in s): return ''.join(str(digits[c]) for c in s)
    units={'十':10,'百':100,'千':1000,'万':10000,'亿':100000000}
    total=section=num=0
    for c in s:
        if c in digits:num=digits[c]
        elif c in units:
            u=units[c]
            if u<10000: section+=(num or 1)*u
            else: total+=(section+num or 1)*u; section=0
            num=0
        else:return s
    return str(total+section+num)
def canonical(s):
    s=norm(s)
    s=re.sub(r'(\d{1,2})点(?=\d{1,2}分|$)',r'\1时',s)
    scales={'万亿':decimal.Decimal('1e12'),'千万':decimal.Decimal('1e7'),'百万':decimal.Decimal('1e6'),'十万':decimal.Decimal('1e5'),'亿':decimal.Decimal('1e8'),'万':decimal.Decimal('1e4'),'千':decimal.Decimal('1e3'),'百':decimal.Decimal('1e2')}
    s=re.sub(r'(\d+(?:\.\d+)?)(万亿|千万|百万|十万|亿|万|千|百)',lambda m:format(decimal.Decimal(m[1])*scales[m[2]],'f').rstrip('0').rstrip('.') if '.' in format(decimal.Decimal(m[1])*scales[m[2]],'f') else format(decimal.Decimal(m[1])*scales[m[2]],'f'),s)
    s=re.sub(r'[零〇一二两三四五六七八九十百千万亿]+',lambda m:han_number(m.group()),s)
    # Keep approximation and inequality qualifiers: 上千 != 数千; 超过 != exact.
    return s
NUM=r'(?:-?\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)'
QUAL=r'(?:超过|超|不足|低于|近|约|上|数|第|零下)?'
SCALE=r'(?:万亿|千万|百万|十万|亿|万|千|百)?'
UNIT=r'(?:亿元|万元|美元|元|%|个百分点|EFlops|dB/100km|dB|℃|公里|小时|分钟|秒|兆|G|T|克|米|亩|人|名|户|家|辆|个|款|条|项|件|份|次|套|路|倍|天|支|部|本|维|量子比特)'
DATE=re.compile(r'(?:\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?|\d{1,2}月\d{1,2}日|\d{1,2}[点时](?:\d{1,2}分)?|\d{1,2}·\d{1,2})')
QUANTITY=re.compile(QUAL+NUM+SCALE+r'(?:余|多)?'+UNIT+r'(?:以上|以下|余|多)?',re.I)
VAGUE=re.compile(r'(?:上|数|近|约)?(?:千余|万余|百余|十余|千|万|百|十)(?:万|亿)?(?:元|人|名|户|家|辆|个|款|条|项|件|次)|一半|八分之一')
NUMBER=re.compile(r'(?<![A-Za-z\d])\d+(?:\.\d+)?(?![A-Za-z\d])')
def extract(text):
    s=norm(text); anchors=[]; occupied=[]
    def add(kind,raw,a,b,method):
        value=canonical(raw) if kind in ['number','percentage','amount','date_year'] else norm(raw).casefold()
        anchors.append({'type':kind,'surface':raw,'normalized':value,'start':a,'end':b,'method':method})
    # Maximal curated name spans. A name can have nested location information only as separate evidence elsewhere.
    terms=sorted([(norm(t),k) for k,ts in LEXICON.items() for t in ts],key=lambda x:(-len(x[0]),x[0]))
    for term,kind in terms:
        for m in re.finditer(re.escape(term),s,re.I):
            if any(m.start()<b and m.end()>a for a,b in occupied):continue
            # Avoid 4G inside an undeclared longer alphanumeric product.
            if term[0].isascii() and m.start()>0 and s[m.start()-1].isascii() and s[m.start()-1].isalnum():continue
            add(kind,m.group(),m.start(),m.end(),'frozen_fact_lexicon');occupied.append(m.span())
    for pattern,forced in [(DATE,'date_year'),(QUANTITY,None),(VAGUE,None),(NUMBER,'number')]:
        for m in pattern.finditer(s):
            if any(m.start()<b and m.end()>a for a,b in occupied):continue
            raw=m.group();kind=forced or ('amount' if '元' in raw else 'percentage' if '%' in raw or '分之一' in raw or raw=='一半' else 'number')
            add(kind,raw,m.start(),m.end(),'numeric_regex');occupied.append(m.span())
    # Open vocabulary named mentions: high recall supplemental flags only, never automatic errors.
    for m in re.finditer(r'[A-Za-z][A-Za-z0-9]*(?:[-+./][A-Za-z0-9]+)*|[“「]([^”」]{2,50})[”」]',s):
        if any(m.start()<b and m.end()>a for a,b in occupied):continue
        raw=m.group(1) or m.group()
        if raw in ['AI','IP','dB','km','EFlops','GDP']:continue
        add('unclassified_named_span',raw,m.start(),m.end(),'open_vocabulary_regex')
    unique={}
    for a in sorted(anchors,key=lambda x:x['start']):
        key=(a['type'],a['normalized'])
        if key not in unique: unique[key]=a
    return list(unique.values())
def compare(facts,text):
    fa=extract('\n'.join(x['fact'] for x in facts));ca=extract(text)
    fd={(x['type'],x['normalized']):x for x in fa};cd={(x['type'],x['normalized']):x for x in ca}
    return {'retained_anchors':[cd[k] for k in fd.keys() & cd.keys()], 'missing_anchors':[fd[k] for k in fd.keys()-cd.keys()], 'new_exact_anchors':[cd[k] for k in cd.keys()-fd.keys()], 'facts_anchor_count':len(fd),'candidate_anchor_count':len(cd)}
def main():
    pack=[json.loads(s) for s in (ROOT/'data/interim/anonymous_candidate_pack.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    for c in pack:
        row={'candidate_id':c['candidate_id'],'version':VERSION,**compare(c['confirmed_facts'],c['candidate_text'])}
        for k in ['retained_anchors','missing_anchors','new_exact_anchors']:row[k]=sorted(row[k],key=lambda a:(a['type'],a['normalized']))
        rows.append(row)
    jsonl(ROOT/'anchor_audit_results.jsonl',rows)
    dump(ROOT/'data/interim/anchor_lexicon.json',{'version':VERSION,'source':'Manually enumerated ONLY from supplied confirmed facts; frozen before stage comparison','lexicon':LEXICON,'limitations':['Names outside frozen fact lexicon detected only for Latin tokens and quoted spans; arbitrary new Chinese proper names may be missed.','Quantities compare surface-normalized anchors, not relations or entailment; qualification differences and range decomposition can create benign missing/new pairs.','No anchor count is a semantic error count. Codes and product numbers are masked from generic numeric regex.','Whole-document retention does not prove correct entity, relation, scope or units.']})
    print(json.dumps({'rows':len(rows),'anchors':sum(x['facts_anchor_count'] for x in rows)},ensure_ascii=False))
if __name__=='__main__':main()

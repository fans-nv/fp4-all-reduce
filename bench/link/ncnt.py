import re,sys,collections
# per-op NVLink bytes for nccl-tests counter brackets: data Tx per GPU / (ops * S), packet size, wire factor
f,ops,S=sys.argv[1],int(sys.argv[2]),float(1<<30)
cur=None;D={};perf={};last=None
for l in open(f):
    m=re.match(r'## CNT (pre|post) (\S+) gpu(\d+)',l)
    if m: cur=(m.group(2),int(m.group(3)),m.group(1)); D[cur]=collections.Counter(); last=m.group(2); continue
    t=l.split()
    if len(t)>=12 and t[0].isdigit(): perf[last]=(float(t[6]),float(t[7]))
    m=re.search(r'Link \d+: (Data Tx|Data Rx|Tx packets|Tx bytes): (\d+)',l)
    if m and cur: D[cur][m.group(1)]+=int(m.group(2))
for n in dict.fromkeys(k[0] for k in D):
    gs=sorted({k[1] for k in D if k[0]==n}); out=[]
    for g in gs:
        a,b=D[(n,g,'pre')],D[(n,g,'post')]; d={k:b[k]-a[k] for k in b}
        tx=d['Data Tx']*1024; out.append((tx/ops/S, d['Tx bytes']/max(1,d['Tx packets']), d['Tx bytes']/max(1,tx), d['Data Rx']*1024/ops/S))
    tx=sum(o[0] for o in out)/len(out); pk=sum(o[1] for o in out)/len(out); wf=sum(o[2] for o in out)/len(out)
    alg,bus=perf.get(n,(0,0))
    print(f"{n:22s} algbw {alg:6.0f} busbw {bus:6.0f} | NVLink Tx data per GPU per op = {tx:.3f} S (Rx {sum(o[3] for o in out)/len(out):.3f} S) | bytes/pkt {pk:5.1f} wire/data {wf:.3f} | measured payload/dir = {tx*alg:.0f} GB/s, wire/dir = {tx*alg*wf:.0f} GB/s")

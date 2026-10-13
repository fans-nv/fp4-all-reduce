import re,sys,json,collections
# parse "## CNT <pre|post> <name> gpuN <ts>" blocks; sum over links per field
cur=None; D={}; holds={}
for l in open(sys.argv[1]):
    m=re.match(r'## CNT (pre|post) (\S+) gpu(\d+) ([\d.]+)',l)
    if m: cur=(m.group(2),int(m.group(3)),m.group(1)); D[cur]=collections.Counter(); D[cur]['ts']=float(m.group(4)); continue
    if l.startswith('{') and '"hold_s"' in l:
        d=json.loads(l); holds[d['tag'].replace('hold_','')]=d
    m=re.search(r'Link (\d+): (Data Tx|Data Rx|Raw Tx|Raw Rx|Tx packets|Tx bytes|Rx packets|Rx bytes): (\d+)',l)
    if m and cur: D[cur][m.group(2)]+=int(m.group(3))
names=sorted({k[0] for k in D}, key=lambda n: list(D).index((n,0,'pre')))
for n in names:
    h=holds.get(n); pay=h['payload_per_dir_bytes'] if h else 0; el=h['hold_s'] if h else 3
    print(f"== {n}: payload/dir {pay/1e9:.2f} GB in {el:.2f}s = {pay/el/1e9 if el else 0:.0f} GB/s; bracket {D[(n,0,'post')]['ts']-D[(n,0,'pre')]['ts']:.2f}s")
    for g in (0,1):
        a=D[(n,g,'pre')]; b=D[(n,g,'post')]; d={k:b[k]-a[k] for k in b if k!='ts'}
        dt=b['ts']-a['ts']
        kib=lambda k: d[k]*1024
        print(f"  gpu{g}: dataTx {kib('Data Tx')/1e9:7.2f} GB dataRx {kib('Data Rx')/1e9:7.2f} rawTx {kib('Raw Tx')/1e9:7.2f} rawRx {kib('Raw Rx')/1e9:7.2f} | raw/data Tx {kib('Raw Tx')/max(1,kib('Data Tx')):.3f} Rx {kib('Raw Rx')/max(1,kib('Data Rx')):.3f} | dataTx/payload {kib('Data Tx')/pay if pay else 0:.3f} | pkts Tx {d['Tx packets']/1e6:.1f}M bytes/pkt Tx {d['Tx bytes']/max(1,d['Tx packets']):.1f} Rx {d['Rx bytes']/max(1,d['Rx packets']):.1f} | Txbytes/data {d['Tx bytes']/max(1,kib('Data Tx')):.3f} | rawTx rate {kib('Raw Tx')/el/1e9:.0f} GB/s")

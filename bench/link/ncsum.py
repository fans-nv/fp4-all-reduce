import sys,os,re,glob
# nccl-tests summary: out-of-place busbw (GB/s) at selected sizes, max over sizes, size reaching 95% of max
sel=[1<<20,16<<20,64<<20,256<<20,1<<30,8<<30]
print(f"{'cell':44s}"+''.join(f"{(s>>20):>7d}M" for s in sel)+"   max  @95%  algbw@max")
for d in sys.argv[1:]:
  for f in sorted(glob.glob(d+'/*.txt')):
    rows={}
    for l in open(f):
        t=l.split()
        if len(t)>=12 and t[0].isdigit():
            try: rows[int(t[0])]=(float(t[-6]) if False else float(t[6]), float(t[7]))
            except: pass
    if not rows:
        err=[l.strip() for l in open(f) if 'rror' in l or 'rc=' in l][-1:]
        print(f"{os.path.basename(f)[:-4]:44s} NO DATA {err}"); continue
    mx=max(rows,key=lambda s:rows[s][1]); m=rows[mx][1]
    s95=min(s for s in rows if rows[s][1]>=0.95*m)
    print(f"{os.path.basename(f)[:-4]:44s}"+''.join(f"{rows[s][1]:8.0f}" if s in rows else '       -' for s in sel)+f" {m:6.0f} {s95>>20:5d}M {rows[mx][0]:6.0f}")

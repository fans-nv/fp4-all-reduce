#!/usr/bin/env python3
"""Parse nccl-tests sweep logs -> table of out-of-place time per (config, size); best per size."""
import sys, os, re, glob, collections
d = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else 'ar0'
rows = collections.defaultdict(dict)  # tag -> size -> (oop, inp)
for f in sorted(glob.glob(f'{d}/{prefix}_*.log')):
    tag = os.path.basename(f)[len(prefix)+1:-4]
    for line in open(f):
        p = line.split()
        if len(p) >= 13 and p[0].isdigit():
            try: rows[tag][int(p[0])] = (float(p[5]), float(p[9]))
            except ValueError: pass
sizes = sorted({s for t in rows.values() for s in t})
print('tag'.ljust(18) + ''.join(f'{s:>11}' for s in sizes))
for t in sorted(rows, key=lambda t: (t != 'default', t)):
    print(t.ljust(18) + ''.join(f'{rows[t][s][0]:>11.2f}' if s in rows[t] else f'{"-":>11}' for s in sizes))
print('BEST (oop):')
for s in sizes:
    c = sorted((rows[t][s][0], t) for t in rows if s in rows[t])
    print(f'  {s:>11} ' + '  '.join(f'{t}={v:.2f}' for v, t in c[:4]) + f'  | default={rows["default"].get(s,(0,))[0]:.2f}')

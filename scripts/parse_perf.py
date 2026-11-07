import os
import statistics
import sys
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PERF_LOG", "swiglu_perf.log")
rows = []
cur = None
for l in open(LOG):
    l = l.rstrip()
    if l.startswith('Testing SwiGLU'):
        cur = {'s': l.split('shape:(')[1].split(')')[0],
               'd': l.split('dtype:')[1].split()[0],
               't': None, 'l': None}
        rows.append(cur)
    elif 'PyTorch time' in l and cur is not None:
        cur['t'] = float(l.split(':')[1].split()[0])
    elif 'lib time' in l and cur is not None:
        cur['l'] = float(l.split(':')[1].split()[0])


def nelem(s):
    n = 1
    for t in s.split(','):
        n *= int(t.strip())
    return n


g = defaultdict(list)
for r in rows:
    if r['t'] and r['l']:
        g[(r['s'], nelem(r['s']))].append((r['t'], r['l']))
print('shape | elems | torch_ms | lib_ms | speedup | GB/s')
for (s, n), v in sorted(g.items(), key=lambda kv: kv[0][1]):
    t = statistics.median(x[0] for x in v)
    li = statistics.median(x[1] for x in v)
    print('(%s) | %d | %.5f | %.5f | %.2fx | %.1f' % (s, n, t, li, t / li, n * 6 / (li / 1e3) / 1e9))
print('rows', len(rows), 'paired', sum(len(x) for x in g.values()))

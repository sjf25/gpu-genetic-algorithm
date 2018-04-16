#!/usr/bin/python3

import sys
import random

n = int(sys.argv[1])
density = float(sys.argv[2])

target_m = int(n * (n - 1) * density / 2)

'''
edges = set()
while len(edges) < target_m:
    u = random.randint(0, n-2)
    v = random.randint(u, n-1)
    e = (u, v)
    edges.add(e)
'''

edges = set()
for u in range(n):
    for v in range(u, n-1):
        e = (u, v)
        prob = random.random()
        if prob <= density:
            edges.add(e)

print("p edge", n, target_m)
for e in edges:
    u, v = e
    print("e", u, v)

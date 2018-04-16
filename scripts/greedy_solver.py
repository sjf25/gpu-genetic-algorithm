#!/usr/bin/python3

# TODO: finish

import sys
in_file = open(sys.argv[1], 'r')
lines = in_file.readlines()
in_file.close()

_, _, n, m = lines[0].split()
n = int(n)
m = int(m)



for i in range(1, len(lines)):
    raw = lines[i].split()

print(n, m)

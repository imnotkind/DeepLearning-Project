import csv
from fen2word import FenToWord
from pwn import *
import json

with open('gameknot.csv') as f:
    r = csv.reader(f)
    l = list(r)

f2w = FenToWord
for row in l[1:]:
    [fen, move, _] = row[3:]

    r = remote('localhost', 51119)
    data = {'fen': fen, 'move': move, 'num_words': 1}
    r.sendline(json.dumps(data))
    print(r.recvline().strip().decode())
    r.close()
    input()

import hailjwt as hj
import json
import sys

with open(sys.argv[1]) as f:
    c = hj.JWTClient(f.read())

sys.stdout.buffer.write(c.encode(json.loads(sys.stdin.read())))

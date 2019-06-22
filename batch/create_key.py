import gear.auth as hj
import json
import sys

with open(sys.argv[1], 'rb') as f:
    c = hj.JWTClient(f.read())

sys.stdout.write(c.encode(json.loads(sys.stdin.read())))

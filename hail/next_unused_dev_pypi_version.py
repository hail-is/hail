import re
import sys
import json
import distutils.version as duv

data = json.load(sys.stdin)
versions = list(data["releases"])

latest = max(versions, key=duv.LooseVersion)
dev_version = re.compile('.*\.dev([0-9]+)$').match(latest)
if dev_version:
    print(int(dev_version[1]) + 1)
else:
    print('0')

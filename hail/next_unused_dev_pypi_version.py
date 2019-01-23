import re
import sys
import json
import urllib.request as urllib
import distutils.version as duv

url = "https://pypi.python.org/pypi/%s/json" % (sys.argv[1],)
data = json.load(urllib.urlopen(urllib.Request(url)))
versions = list(data["releases"])

latest = max(versions, key=duv.LooseVersion)
dev_version = re.compile('.*\.dev([0-9]+)$').match(latest)
if dev_version:
    print(int(dev_version[1]) + 1)
else:
    print('0')

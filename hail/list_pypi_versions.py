import sys
import json
import urllib.request as urllib
import distutils.version as duv

url = "https://pypi.python.org/pypi/%s/json" % (sys.argv[1],)
data = json.load(urllib.urlopen(urllib.Request(url)))
versions = list(data["releases"])
versions.sort(key=duv.LooseVersion)
print('\n'.join(versions))

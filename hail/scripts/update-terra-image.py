import datetime
import json
import sys

assert len(sys.argv) == 3, sys.argv
hail_pip_version = sys.argv[1]
image_name = sys.argv[2]

with open("config/conf.json") as conf_f:
    conf = json.load(conf_f)

[hail_data] = [data for data in conf["image_data"] if data.get("name") == image_name]
hail_image_version = hail_data["version"]
hail_image_version = [int(n) for n in hail_image_version.split(".")]
hail_image_version[-1] += 1
hail_data["version"] = ".".join(str(i) for i in hail_image_version)
image_version = hail_data["version"]

with open("config/conf.json", "w") as conf_f:
    json.dump(conf, conf_f, indent=4, separators=(",", " : "))
    print(file=conf_f)  # newline at end of file

with open(f'{image_name}/CHANGELOG.md') as fobj:
    changelog = fobj.read()

with open(f'{image_name}/CHANGELOG.md', 'w') as fobj:
    todays_date = datetime.date.today().strftime('%Y-%m-%d')
    fobj.write(f"""## {image_version} - {todays_date}
- Update `hail` to `{hail_pip_version}`
  - See https://hail.is/docs/0.2/change_log.html#version-{hail_pip_version.replace('.', '-')}) for details

Image URL: `us.gcr.io/broad-dsp-gcr-public/{image_name}:{image_version}`

""")
    fobj.write(changelog)


def update_version_line(line):
    if line.startswith('ENV HAIL_VERSION'):
        return 'ENV HAIL_VERSION=' + hail_pip_version
    return line


with open(f'{image_name}/Dockerfile') as fobj:
    dockerfile = fobj.read()

with open(f'{image_name}/Dockerfile', 'w') as fobj:
    fobj.write('\n'.join([update_version_line(line) for line in dockerfile.split('\n')]))

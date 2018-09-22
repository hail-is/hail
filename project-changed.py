import sys
import subprocess

if len(sys.argv) != 3:
    sys.stderr.write(f'''usage: {sys.argv[0]} <orig-hash> <project>

outputs 'yes' if <project> changed in HEAD compared to <orig-hash> else 'no'.
''')
    exit(1)

# TODO dependencies
projects = {
    'hail': 'hail/',
    'batch': 'batch/',
    'ci': 'ci/',
    'site': 'site/',
    'scorecard': 'scorecard/'
}

orig_hash = sys.argv[1]
target_project = sys.argv[2]

if target_project not in projects:
    sys.stderr.write(f'unknown project: {target_project}\n')
    exit(1)

cmd = ['git', 'diff', '--name-only', orig_hash]
proc = subprocess.run(cmd, stdout=subprocess.PIPE, encoding='utf-8')
if proc.returncode != 0:
    sys.stderr.write(f"command exited with return code {proc.returncode}: {' '.join(cmd)}")
    exit(1)

def get_project(line):
    for project, prefix in projects.items():
        if line.startswith(prefix):
            return project
    return None

for line in proc.stdout.split('\n'):
    line = line.strip()
    if not line:
        continue

    line_project = get_project(line)
    if line_project == target_project or line_project is None:
        print('yes')
        exit(0)

print('no')
exit(0)

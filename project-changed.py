import sys
import subprocess

if len(sys.argv) != 3:
    sys.stderr.write('''usage: {} <orig-hash> <project>

outputs 'yes' if <project> changed in HEAD compared to <orig-hash> else 'no'.
    ''', sys.argv[0])
    exit(1)

projects = []
with open('projects.txt', 'r') as f:
    for  line in f:
        line = line.strip()
        if not line:
            continue

        projects.append(line)

orig_hash = sys.argv[1]
target_project = sys.argv[2]

if target_project not in projects:
    sys.stderr.write('unknown project: {}\n'.format(target_project))
    exit(1)

cmd = ['git', 'diff', '--name-only', orig_hash]
proc = subprocess.run(cmd, stdout=subprocess.PIPE)
if proc.returncode != 0:
    sys.stderr.write('command exited with return code {}: {}'.format(proc.returncode, ' '.join(cmd)))
    exit(1)

def get_project(line):
    for project in projects:
        if line.startswith(project + '/'):
            return project
    return None

for line in proc.stdout.decode('utf-8').split('\n'):
    line = line.strip()
    if not line:
        continue

    line_project = get_project(line)
    if line_project == target_project or line_project is None:
        print('yes')
        exit(0)

print('no')
exit(0)

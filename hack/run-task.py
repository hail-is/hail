from shlex import quote as shq
import time
import json
import subprocess
import requests

with open('/config.json') as f:
    config = json.loads(f.read())

task_token = config['task_token']
inst_token = config['inst_token']
scratch_dir = config['scratch_dir']
inputs_cmd = config['inputs_cmd']
image = config['image']
cmd = config['command']
outputs_cmd = config['outputs_cmd']
master = config['master']

def check_shell(script):
    subprocess.check_call(['/bin/bash', '-c', script])

def check_shell_output(script):
    return subprocess.check_output(['/bin/bash', '-c', script], encoding='utf-8')

def docker_run(name, image, cmd):
    container_id = None
    attempts = 0
    while not container_id:
        try:
            container_id = check_shell_output(f'docker run -d -v /shared:/shared {shq(image)} /bin/bash -c {shq(cmd)}').strip()
        except subprocess.CalledProcessError as e:
            if attempts < 12 and e.returncode == 125:
                attempts += 1
                time.sleep(5)
            else:
                raise e

    ec_str = check_shell_output(f'docker container wait {shq(container_id)}').strip()
    ec = int(ec_str)

    print(f'ec {name} {ec}')

    local_log = f'{name}.log'
    gs_log = f'{scratch_dir}/{inst_token}/{name}.log'

    check_shell(f'docker logs {container_id} > {shq(local_log)} 2>&1')
    check_shell(f'gsutil cp {shq(local_log)} {shq(gs_log)}')

    return ec

# credentials
input_ec = docker_run('input', 'google/cloud-sdk:237.0.0-alpine', inputs_cmd)

status = {
    'task_token': task_token,
    'inst_token': inst_token,
    'input': input_ec
}

if input_ec == 0:
    main_ec = docker_run('main', image, cmd)
    status['main'] = main_ec

    if main_ec == 0:
        output_ec = docker_run('output', 'google/cloud-sdk:237.0.0-alpine', outputs_cmd)
        status['output'] = output_ec

print(f'status {status}')

requests.post(f'http://{master}:5000/status', json=status)

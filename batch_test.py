import hailtop.batch as hb
import subprocess

backend = hb.ServiceBackend(
        billing_project='nnf-karczewski',
        regions=['us-central1']
)

b = hb.Batch(
    'deep-learning',
    backend=backend,
)


def chk_cuda():
    subprocess.run("pip install torch", shell=True)
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

j = b.new_python_job(name='chk-cuda')
j._machine_type = "g2-standard-24"
j.storage('20Gi')
j.output1 = j.call(chk_cuda)
b.run()

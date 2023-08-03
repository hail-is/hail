import hailtop.batch as hb
backend = hb.ServiceBackend('nnfc-fdp', remote_tmpdir='gs://nnfc-fdp-tmp-7day/parsa/tmp')
b = hb.Batch(backend=backend, name='test')
j = b.new_job(name='sophie-test')
j.image("us-central1-docker.pkg.dev/nnfc-fdp/gpu-image/gpu")
j._preemptible = False
j.command("python3 -c 'import torch; print(torch.cuda.is_available())'")
#j.command("python3 -c 'import time; time.sleep(3600)'")
j._machine_type = "g2-standard-4"
j.max_live_instances = 1
j.storage('100Gi')
b.run()

import client
import time

batch = client.BatchClient('http://localhost:5000')

sleep = batch.create_batch('sleep')

j = sleep.create_job('clitest', 'alpine', ['/bin/sh', '-c', 'sleep 5 && echo hi'])
print(j)
j2 = sleep.create_job('clitest', 'alpine', ['/bin/sh', '-c', 'sleep 3 && echo there'])
j3 = sleep.create_job('clitest', 'alpine', ['/bin/sh', '-c', 'sleep 7 && echo you'])

print(sleep.status())

sleep.wait()
print(sleep.status())

print(j2.status())

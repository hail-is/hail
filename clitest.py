import client
import time

client.configure('http://localhost:5000')

j = client.create_job('clitest', 'alpine', ['/bin/sh', '-c', 'sleep 5 && echo hi'])
print(j)

time.sleep(2)

client.cancel(j['id'])
j = client.get_job(j['id'])
print(j)

# j = client.wait(j['id'])
# print(j)

import batch.client
c = batch.client.BatchClient('http://localhost:5000')
j = c.create_job('gcr.io/broad-ctsa/batch-test')
status = j.wait()
print(status)

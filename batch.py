import threading
import asyncio
import aiohttp
from aiohttp import web
import kubernetes as kube

# FIXME periodically scan all pods?
# FIXME test killing node
# FIXME schedule on preemptibles

# FIXME bioinformatics images, set up gsutil
# FIXME mark completed pods as resolved, 

kube.config.load_kube_config()
v1 = kube.client.CoreV1Api()

pod_job = {}
name_job = {}

counter = 0
def next_uid():
    global counter
    
    counter = counter + 1
    return counter

def create_pod(name, image):
    pod = kube.client.V1Pod(
        metadata = kube.client.V1ObjectMeta(generate_name = name + '-'),
        spec = kube.client.V1PodSpec(
            containers = [
                kube.client.V1Container(
                    name = 'default',
                    image = image)
            ],
            restart_policy = 'Never'))
    res = v1.create_namespaced_pod('default', pod)
    return res.metadata.uid

class Job(object):
    def __init__(self, name, image):
        self.name = name
        self.uid = next_uid()
        self.image = image
        self.state = 'Created'
        print('created job {}'.format(self.uid))

        name_job[self.name] = self

    def set_state(self, new_state):
        if self.state != new_state:
            print('job {} changed state: {} -> {}'.format(
                self.uid,
                self.state,
                new_state))
            self.state = new_state

    def schedule(self):
        uid = create_pod(self.name, self.image)
        pod_job[uid] = self
        self.set_state('Scheduling')

    def mark_scheduled(self):
        self.set_state('Scheduled')

    def mark_unscheduled(self):
        uid = create_pod(self.name, self.image)
        pod_job[uid] = self
        self.set_state('Scheduling')

    def mark_complete(self, exit_code):
        self.set_state('Complete exit_code: {}'.format(exit_code))
        self.exit_code = exit_code


async def schedule(request):
    parameters = await request.json()
    name = parameters['name']
    image = parameters['image']
    if name in name_job:
        return web.Response(text='job {} already exists'.format(name), status=400)
    job = Job(name, image)
    job.schedule()
    result = {'uid': job.uid, 'name': job.name}
    return web.json_response(result)

def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = web.Application()
    app.router.add_routes([web.get('/schedule', schedule)])
    web.run_app(app, host='localhost', port=8080)

def kube_event_loop():
    stream = kube.watch.Watch().stream(v1.list_namespaced_pod, 'default')
    for event in stream:
        # print(event)
        event_type = event['type']

        pod = event['object']
        name = pod.metadata.name
        uid = pod.metadata.uid

        job = pod_job.get(uid)
        if job and job.state != 'Complete':
            if event_type == 'DELETE':
                job.mark_unscheduled()
                del job[uid]
            else:
                assert event_type == 'ADDED' or event_type == 'MODIFIED'
                if pod.status.container_statuses:
                    assert len(pod.status.container_statuses) == 1
                    container_status = pod.status.container_statuses[0]
                    assert container_status.name == 'default'

                    if container_status.state and container_status.state.terminated:
                        exit_code = container_status.state.terminated.exit_code
                        job.mark_complete(exit_code)
                    else:
                        job.mark_scheduled()
                else:
                    job.mark_scheduled()

t = threading.Thread(target=kube_event_loop)
t.start()

# asyncio has to go in main thread
run()

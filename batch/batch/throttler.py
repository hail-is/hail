import asyncio
import logging
import traceback


log = logging.getLogger('batch.throttler')


class PodThrottler:
    def __init__(self, queue_size, max_pods, parallelism=1):
        self.queue_size = queue_size
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.semaphore = asyncio.BoundedSemaphore(max_pods)
        self.pending_pods = set()
        self.created_pods = set()

        workers = [asyncio.ensure_future(self._create_pod())
                   for _ in range(parallelism)]

        async def manager(workers):
            while True:
                failed, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)
                for fut in failed:
                    err = fut.exception()
                    assert err is not None
                    err_msg = '\n'.join(
                        traceback.format_exception(type(err), err, err.__traceback__))
                    log.error(f'restarting failed worker: {err} {err_msg}')
                    pending.add(asyncio.ensure_future(self._create_pod()))
                workers = pending

        asyncio.ensure_future(manager(workers))

    async def _create_pod(self):
        while True:
            await self.semaphore.acquire()
            try:
                job = await self.queue.get()
                pod_name = job._pod_name

                if pod_name not in self.pending_pods:
                    log.info(f'pod {pod_name} was deleted before it was created, ignoring')
                    self.semaphore.release()
                    return

                await job._create_pod()
            except:
                self.semaphore.release()
                raise

            self.pending_pods.remove(pod_name)
            self.created_pods.add(pod_name)
            self.queue.task_done()

    def is_queued(self, job):
        return job._pod_name in self.pending_pods

    def create_pod(self, job):
        # this method does not wait for the pod to be created before returning
        pod_name = job._pod_name

        if pod_name in self.pending_pods or pod_name in self.created_pods:
            log.info(f'job {job.id} is already in the queue, ignoring')
            return None

        self.pending_pods.add(pod_name)

        try:
            self.queue.put_nowait(job)
            return None
        except asyncio.QueueFull as err:
            self.pending_pods.remove(pod_name)
            log.info(f'pod queue full, could not create {pod_name}')
            return err

    async def delete_pod(self, job):
        await job._delete_pod()
        pod_name = job._pod_name
        if pod_name in self.pending_pods:
            self.pending_pods.remove(pod_name)
            log.info(f'deleted pending pod {job.id}')
        elif pod_name in self.created_pods:
            self.created_pods.remove(pod_name)
            self.semaphore.release()

    def full(self):
        return self.queue.full()

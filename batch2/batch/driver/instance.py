import time
import logging
import googleapiclient.errors
import asyncio
import aiohttp

log = logging.getLogger('instance')


class Instance:
    @staticmethod
    def from_record(inst_pool, record):
        ip_address = record['ip_address']

        pending = ip_address is None
        active = ip_address is not None
        deleted = False

        inst = Instance(inst_pool, record['name'], record['token'],
                        ip_address=ip_address, pending=pending,
                        active=active, deleted=deleted)

        inst_pool.free_cores_mcpu += inst_pool.worker_capacity_mcpu  # FIXME: this should get cores from db in future

        if active:
            inst_pool.n_active_instances += 1
            inst_pool.instances_by_free_cores.add(inst)
        else:
            assert pending
            inst_pool.n_pending_instances += 1

        log.info(f'added instance {inst.name} to the instance pool with ip address {inst.ip_address}')

        return inst

    @staticmethod
    async def create(inst_pool, name, token):
        # FIXME: maybe add machine type, cores, batch_image etc.
        await inst_pool.driver.db.instances.new_record(name=name,
                                                       token=token)

        inst_pool.n_pending_instances += 1
        inst_pool.free_cores_mcpu += inst_pool.worker_capacity_mcpu

        return Instance(inst_pool, name, token, ip_address=None, pending=True,
                        active=False, deleted=False)

    def __init__(self, inst_pool, name, token, ip_address, pending, active, deleted):
        self.inst_pool = inst_pool
        self.name = name
        self.token = token
        self.ip_address = ip_address

        self.lock = asyncio.Lock()

        self.pods = set()
        self.free_cores_mcpu = inst_pool.worker_capacity_mcpu

        # state: pending, active, deactivated (and/or deleted)
        self.pending = pending
        self.active = active
        self.deleted = deleted

        self.healthy = True
        self.last_updated = time.time()
        self.time_created = time.time()
        self.last_ping = time.time()

        log.info(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')

    def unschedule(self, pod):
        assert not self.pending and self.active
        self.pods.remove(pod)

        if self.healthy:
            self.inst_pool.instances_by_free_cores.remove(self)
            self.free_cores_mcpu += pod.cores_mcpu
            self.inst_pool.free_cores_mcpu += pod.cores_mcpu
            self.inst_pool.instances_by_free_cores.add(self)
            self.inst_pool.driver.changed.set()
        else:
            self.free_cores_mcpu += pod.cores_mcpu

    def schedule(self, pod):
        assert not self.pending and self.active and self.healthy
        self.pods.add(pod)
        self.inst_pool.instances_by_free_cores.remove(self)
        self.free_cores_mcpu -= pod.cores_mcpu
        self.inst_pool.free_cores_mcpu -= pod.cores_mcpu
        assert self.inst_pool.free_cores_mcpu >= 0, (self.inst_pool.free_cores_mcpu, pod.cores_mcpu)
        self.inst_pool.instances_by_free_cores.add(self)
        # can't create more scheduling opportunities, don't set changed

    async def activate(self, ip_address):
        async with self.lock:
            log.info(f'activating instance {self.name} after {time.time() - self.time_created} seconds since creation')
            if self.active:
                return
            if self.deleted:
                return

            if self.pending:
                self.pending = False
                self.inst_pool.n_pending_instances -= 1
                self.inst_pool.free_cores_mcpu -= self.inst_pool.worker_capacity_mcpu

            self.active = True
            self.ip_address = ip_address
            self.inst_pool.n_active_instances += 1
            self.inst_pool.instances_by_free_cores.add(self)
            self.inst_pool.free_cores_mcpu += self.inst_pool.worker_capacity_mcpu
            self.inst_pool.driver.changed.set()

            await self.inst_pool.driver.db.instances.update_record(
                self.name, ip_address=ip_address)

            log.info(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')

    async def deactivate(self):
        async with self.lock:
            log.info(f'deactivating instance {self.name}')
            start = time.time()
            if self.pending:
                self.pending = False
                self.inst_pool.n_pending_instances -= 1
                self.inst_pool.free_cores_mcpu -= self.inst_pool.worker_capacity_mcpu
                assert not self.active
                log.info(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')
                return

            if not self.active:
                return

            self.mark_as_unhealthy()

            pod_list = list(self.pods)
            await asyncio.gather(*[p.unschedule() for p in pod_list])
            assert not self.pods

            for pod in pod_list:
                asyncio.ensure_future(pod.put_on_ready())

            self.active = False

            log.info(f'took {time.time() - start} seconds to deactivate {self.name}')
            log.info(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')

    def update_timestamp(self):
        if self in self.inst_pool.instances:
            self.inst_pool.instances.remove(self)
            self.last_updated = time.time()
            self.inst_pool.instances.add(self)

    def mark_as_unhealthy(self):
        if not self.active or not self.healthy:
            return

        self.inst_pool.instances.remove(self)
        self.healthy = False
        self.inst_pool.instances.add(self)

        if self in self.inst_pool.instances_by_free_cores:
            self.inst_pool.instances_by_free_cores.remove(self)
            self.inst_pool.n_active_instances -= 1
            self.inst_pool.free_cores_mcpu -= self.free_cores_mcpu

        self.update_timestamp()

    def mark_as_healthy(self):
        self.last_ping = time.time()

        if not self.active or self.healthy:
            return

        self.inst_pool.instances.remove(self)
        self.healthy = True
        self.inst_pool.instances.add(self)

        if self not in self.inst_pool.instances_by_free_cores:
            self.inst_pool.n_active_instances += 1
            self.inst_pool.instances_by_free_cores.add(self)
            self.inst_pool.free_cores_mcpu += self.free_cores_mcpu
            self.inst_pool.driver.changed.set()

    async def remove(self):
        log.info(f'removing instance {self.name}')
        await self.deactivate()
        self.inst_pool.instances.remove(self)
        if self.token in self.inst_pool.token_inst:
            del self.inst_pool.token_inst[self.token]
        await self.inst_pool.driver.db.instances.delete_record(self.name)

    async def handle_call_delete_event(self):
        log.info(f'handling call delete event for {self.name}')
        await self.deactivate()
        self.deleted = True
        self.update_timestamp()

    async def delete(self):
        log.info(f'deleting instance {self.name}')
        if self.deleted:
            return
        await self.deactivate()
        try:
            await self.inst_pool.driver.gservices.delete_instance(self.name)
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                log.info(f'instance {self.name} was already deleted')
            else:
                raise e
        self.deleted = True

    async def handle_preempt_event(self):
        log.info(f'handling preemption event for {self.name}')
        await self.delete()
        self.update_timestamp()

    async def heal(self):
        log.info(f'healing instance {self.name}')

        async def _heal_gce():
            try:
                spec = await self.inst_pool.driver.gservices.get_instance(self.name)
            except googleapiclient.errors.HttpError as e:
                if e.resp['status'] == '404':
                    await self.remove()
                    return

            status = spec['status']
            log.info(f'heal gce: machine {self.name} status {status}')

            # preempted goes into terminated state
            if status == 'TERMINATED' and self.deleted:
                log.info(f'instance {self.name} is terminated and deleted, removing')
                await self.remove()
                return

            if status in ('TERMINATED', 'STOPPING'):
                log.info(f'instance {self.name} is {status}, deactivating')
                await self.deactivate()

            if status == 'TERMINATED' and not self.deleted:
                log.info(f'instance {self.name} is {status} and not deleted, deleting')
                await self.delete()

            if status == 'RUNNING' and self.active and not self.healthy and time.time() - self.last_ping > 60 * 5:
                log.info(f'instance {self.name} is {status} and not healthy and last ping was greater than 5 minutes, deleting')
                await self.delete()

            if (status in ('STAGING', 'RUNNING')) and not self.active and time.time() - self.time_created > 60 * 5:
                log.info(f'instance {self.name} is {status} and not active and older than 5 minutes, deleting')
                await self.delete()

            self.update_timestamp()

        if self.ip_address and self.active:
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                    await session.get(f'http://{self.ip_address}:5000/healthcheck')
                    self.mark_as_healthy()
                    self.update_timestamp()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as err:  # pylint: disable=broad-except
                log.info(f'healthcheck failed for {self.name} due to err {err}; asking gce instead')
                self.mark_as_unhealthy()
                await _heal_gce()
        else:
            await _heal_gce()

    def __str__(self):
        return self.name

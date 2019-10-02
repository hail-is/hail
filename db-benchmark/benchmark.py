import os
import json
import time
import statistics
import asyncio
import aiomysql

from hailtop.utils import AsyncWorkerPool

config_file = '/secret/sql-config.json'


userdata = {'username': 'jigold',
            'service-account': 'foo@gmail.com',
            'k8s_service_account': 'dfsfasdads@k8s.com'}


# Connect to the database
with open(config_file, 'r') as f:
    config = json.loads(f.read().strip())


def new_record_template(table_name, *field_names):
    names = ", ".join([f'`{name.replace("`", "``")}`' for name in field_names])
    values = ", ".join([f"%({name})s" for name in field_names])
    sql = f"INSERT INTO `{table_name}` ({names}) VALUES ({values})"
    return sql


async def insert_batch(pool, **data):
    async with pool.acquire() as conn:
        start = time.time()
        async with conn.cursor() as cursor:
            sql = new_record_template('batch', *data)
            await cursor.execute(sql, data)
            id = cursor.lastrowid  # This returns 0 unless an autoincrement field is in the table
            return id, time.time() - start


async def insert_jobs(pool, data):
    async with pool.acquire() as conn:
        start = time.time()
        async with conn.cursor() as cursor:
            sql = new_record_template('jobs', *(data[0]))
            await cursor.executemany(sql, data)
        return time.time() - start


def job_spec(image, env=None, command=None, args=None, ports=None,
             resources=None, volumes=None, tolerations=None,
             security_context=None, service_account_name=None):
    if env:
        env = [{'name': k, 'value': v} for (k, v) in env.items()]
    else:
        env = []
    env.extend([{
        'name': 'POD_IP',
        'valueFrom': {
            'fieldRef': {'fieldPath': 'status.podIP'}
        }
    }, {
        'name': 'POD_NAME',
        'valueFrom': {
            'fieldRef': {'fieldPath': 'metadata.name'}
        }
    }])

    container = {
        'image': image,
        'name': 'main'
    }
    if command:
        container['command'] = command
    if args:
        container['args'] = args
    if env:
        container['env'] = env
    if ports:
        container['ports'] = [{
            'containerPort': p,
            'protocol': 'TCP'
        } for p in ports]
    if resources:
        container['resources'] = resources
    if volumes:
        container['volumeMounts'] = [v['volume_mount'] for v in volumes]
    spec = {
        'containers': [container],
        'restartPolicy': 'Never'
    }
    if volumes:
        spec['volumes'] = [v['volume'] for v in volumes]
    if tolerations:
        spec['tolerations'] = tolerations
    if security_context:
        spec['securityContext'] = security_context
    if service_account_name:
        spec['serviceAccountName'] = service_account_name

    return spec


async def test(n_replicates, batch_sizes, parallelism, chunk_size):
    pool = await aiomysql.create_pool(
        host=config['host'],
        port=config['port'],
        db=config['db'],
        user=config['user'],
        password=config['password'],
        charset='utf8',
        cursorclass=aiomysql.cursors.DictCursor,
        autocommit=True,
        maxsize=1000,
        connect_timeout=100)

    try:
        insert_batch_timings = []
        insert_jobs_timings = {}

        for n_jobs in batch_sizes:
            print(n_jobs)
            insert_jobs_timings[n_jobs] = []

            for i in range(n_replicates):
                data = {'userdata': json.dumps(userdata),
                        'user': 'jigold',
                        'deleted': False,
                        'cancelled': False,
                        'closed': False,
                        'n_jobs': 5}
                batch_id, timing = await insert_batch(pool, **data)
                insert_batch_timings.append(timing)

                jobs_data = []
                for job_id in range(n_jobs):
                    spec = job_spec('alpine', command=['sleep', '40'])
                    data = {'batch_id': batch_id,
                            'job_id': job_id,
                            'state': 'Running',
                            'pvc_size': '100M',
                            'callback': 'http://foo.com/callback',
                            'attributes': json.dumps({'job': f'{job_id}',
                                                      'app': 'batch'}),
                            'always_run': False,
                            'token': 'dfsafdsasd',
                            'directory': f'gs://fdsaofsaoureqr/fadsfafdsfdasfd/{batch_id}/{job_id}/',
                            'pod_spec': json.dumps(spec),
                            'input_files': json.dumps(['foo.txt', 'gs://ffadsfe/fadsfad/foo.txt']),
                            'output_files': json.dumps(['gs://fdsfadfefadf/afdsasfewa/wip/a.txt']),
                            'exit_codes': json.dumps([None, None, None])}
                    jobs_data.append(data)

                jobs_data = [jobs_data[i:i + chunk_size] for i in range(0, len(jobs_data), chunk_size)]

                async_pool = AsyncWorkerPool(parallelism)
                start = time.time()

                for jd in jobs_data:
                    await async_pool.call(insert_jobs, pool, jd)

                await async_pool.wait()

                elapsed_time = time.time() - start
                print(f'inserted {n_jobs} jobs in {round(elapsed_time, 3)} seconds')
                insert_jobs_timings[n_jobs].append(elapsed_time)

        for n_jobs, timings in insert_jobs_timings.items():
            print(f'insert {n_jobs} jobs: n={n_replicates} mean={statistics.mean(timings)} stddev={statistics.stdev(timings) if len(timings) > 1 else None}')

    finally:
        pool.close()

n_replicates = int(os.environ.get('N_REPLICATES', 10))
batch_sizes = [int(size) for size in os.environ.get('BATCH_SIZES', '1000,10000,100000').split(",")]
parallelism = int(os.environ.get('PARALLELISM', 1))
chunk_size = int(os.environ.get('CHUNK_SIZE', 1000))

print(f'N_REPLICATES={n_replicates}')
print(f'BATCH_SIZES={batch_sizes}')
print(f'PARALLELISM={parallelism}')
print(f'CHUNK_SIZE={chunk_size}')

loop = asyncio.get_event_loop()
loop.run_until_complete(test(n_replicates=n_replicates,
                             batch_sizes=batch_sizes,
                             parallelism=parallelism,
                             chunk_size=chunk_size))
loop.close()

import asyncio
import os

from hailtop.utils.rates import rate_instance_hour_to_fraction_msec, rate_cpu_hour_to_mcpu_msec
from gear import Database, transaction


async def main():
    cloud = os.environ['HAIL_CLOUD']

    db = Database()
    await db.async_init()

    await db.execute_update('ALTER TABLE `regions` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.execute_update('ALTER TABLE `pools` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.execute_update('ALTER TABLE `instances` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.execute_update('ALTER TABLE `user_inst_coll_resources` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.execute_update('ALTER TABLE `job_groups_inst_coll_staging` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.execute_update('ALTER TABLE `job_group_inst_coll_cancellable_resources` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.execute_update('ALTER TABLE `jobs` ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT %s, ALGORITHM=INSTANT;',
                            (cloud,))

    await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

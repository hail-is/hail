import os
import asyncio
from gear import Database


async def main():
    default_namespace = os.environ['HAIL_NAMESPACE']

    # This fix is only necessary for dev/test namespaces
    if default_namespace == 'default':
        return

    db = Database()
    await db.async_init()

    await db.just_execute(
        '''
DELETE FROM `active_namespaces` WHERE `namespace` = 'default';
INSERT INTO `active_namespaces` (`namespace`) VALUES (%s);
INSERT INTO `deployed_services` (`namespace`, `service`) VALUES
(%s, 'auth'), (%s, 'batch'), (%s, 'batch-driver'), (%s, 'ci');
''',
        (default_namespace,
         default_namespace, default_namespace, default_namespace, default_namespace),
    )
    await db.async_close()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

import asyncio
from gear import Database


class Resource:
    @staticmethod
    def from_record(record):
        return Resource(record['resource_id'], record['resource'], record['rate'])

    def __init__(self, resource_id, resource, rate):
        self.resource_id = resource_id
        self.resource = resource
        product, version = resource.rsplit('/', maxsplit=1)
        self.product = product
        self.version = version
        self.rate = rate


class Product:
    def __init__(self, product):
        self.product = product
        self.latest_resource = None
        self.resource_id_updates = {}

    def add_resource(self, resource: Resource):
        if self.latest_resource is None or self.latest_resource.rate != resource.rate:
            self.resource_id_updates[resource.resource_id] = resource.resource_id
            self.latest_resource = resource
        else:
            self.resource_id_updates[resource.resource_id] = self.latest_resource.resource_id


async def main():
    db = Database()
    try:
        await db.async_init()

        resources = [Resource.from_record(record) async for record in db.execute_and_fetchall('SELECT * FROM resources WHERE deduped_resource_id IS NULL ORDER BY resource_id ASC FOR UPDATE')]

        products = {}
        for resource in resources:
            if resource.product not in products:
                products[resource.product] = Product(resource.product)
            product = products[resource.product]
            product.add_resource(resource)

        resource_id_mapping = {}
        for product in products.values():
            resource_id_mapping.update(product.resource_id_updates)

        update_queries = []
        update_query_args = []
        for resource_id, deduped_resource_id in resource_id_mapping.items():
            update_queries.append('UPDATE resources SET deduped_resource_id = %s WHERE resource_id = %s;')
            update_query_args += [deduped_resource_id, resource_id]

        await db.execute_update(' '.join(update_queries), update_query_args)

        new_ids = [record['deduped_resource_id'] async for record in db.execute_and_fetchall('SELECT * FROM resources')]
        assert all(deduped_resource_id is not None for deduped_resource_id in new_ids)
    finally:
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

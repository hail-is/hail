from .batch_cli_utils import make_formatter


def list(query, limit, before, full, output):
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch_list = client.list_batches(q=query, last_batch_id=before, limit=limit)
        statuses = [batch.last_known_status() for batch in batch_list]

    if len(statuses) == 0:
        print("No batches to display.")
        return

    for status in statuses:
        status['state'] = status['state'].capitalize()

    if full:
        statuses = [{k: v for k, v in status.items() if k != 'attributes'} for status in statuses]
    else:
        statuses = [{'id': status['id'], 'state': status['state']} for status in statuses]

    format = make_formatter(output)
    print(format(statuses))

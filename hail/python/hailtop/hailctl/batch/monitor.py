from typing import List, Optional

from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class BatchOverviewTable:
    def __init__(self,
                 batch_client,
                 include_cluster_stats: bool,
                 include_batch_progress: bool,
                 batch_ids: Optional[List[int]] = None,
                 limit: int = 10):
        from hailtop.utils.rich_widgets import BatchStatusProgress, ClusterCapacityProgress  # pylint: disable=import-outside-toplevel

        self.batch_client = batch_client
        self.progress_table = Table.grid()

        if include_cluster_stats:
            self.cluster_capacity_progress = ClusterCapacityProgress(batch_client)
            self.progress_table.add_row(
                Panel.fit(
                    self.cluster_capacity_progress._progress,
                    title="[b]Cluster Capacity",
                    border_style="black",
                    padding=(1, 2)
                ),
            )
        else:
            self.cluster_capacity_progress = None

        if include_batch_progress:
            self.batch_progress = BatchStatusProgress(batch_client, batch_ids, limit)
            if batch_ids is None:
                title = '[b]Active Batches'
            else:
                title = '[b]Selected Batches'
            self.progress_table.add_row(
                Panel.fit(self.batch_progress._progress, title=title, border_style="black", padding=(1, 2)),
            )
        else:
            self.batch_progress = None

        self._live = Live(self.progress_table)

    async def update(self) -> bool:
        changed = False
        if self.cluster_capacity_progress is not None:
            changed |= await self.cluster_capacity_progress.update()
        if self.batch_progress is not None:
            changed |= await self.batch_progress.update()
        self._live.update(self.progress_table, refresh=True)
        return changed

    async def __aenter__(self):
        if self.cluster_capacity_progress is not None:
            await self.cluster_capacity_progress.initialize()
        if self.batch_progress is not None:
            await self.batch_progress.initialize()
        self._live.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._live.__exit__(exc_type, exc_val, exc_tb)


async def async_monitor(include_cluster_stats: bool = True,
                        include_batch_progress: bool = True,
                        batch_ids: Optional[List[int]] = None,
                        limit: int = 10):
    from hailtop.batch_client.aioclient import BatchClient  # pylint: disable=import-outside-toplevel
    from hailtop.utils import sleep_before_try  # pylint: disable=import-outside-toplevel

    async with await BatchClient.create('') as client:
        async with BatchOverviewTable(client, include_cluster_stats, include_batch_progress, batch_ids, limit) as table:
            while True:
                tries = 0
                while True:
                    changed = await table.update()
                    tries += 1
                    await sleep_before_try(tries, base_delay_ms=5_000, max_delay_ms=10_000)
                    if changed:
                        break

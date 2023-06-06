import collections
from collections import namedtuple
import datetime
from enum import Enum
import functools
from typing import Dict, List, Optional

from rich.color import Color
from rich.progress import (
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.style import Style
from rich.text import Text

from ...batch_client.aioclient import Batch
from ...config import get_deploy_config
from ..rich_multistate_progress_bar import MultiStateProgressColumn, MultiStateProgress
from ..utils import bounded_gather
from .common import StateData


JobStyle = namedtuple('JobStyle', ['label', 'style'])


class JobState(Enum):
    SUCCEEDED = JobStyle('succeeded', Style(color='green'))
    FAILED = JobStyle('failed', Style(color='red'))
    CANCELLED = JobStyle('cancelled', Style(color='yellow'))
    RUNNING = JobStyle('running', Style(color='blue'))
    READY = JobStyle('ready', Style(color='white'))
    CREATING = JobStyle('creating', Style(color='cyan'))

    @staticmethod
    def is_complete_state(state: 'JobState'):
        return state in (JobState.FAILED, JobState.SUCCEEDED, JobState.CANCELLED)


class JobStats:
    @staticmethod
    def from_batch_status(status: dict) -> 'JobStats':
        return JobStats(status['n_jobs'],
                        status['n_succeeded'],
                        status['n_failed'],
                        status['n_cancelled'],
                        status['n_running'],
                        status['n_ready'],
                        status['n_creating'])

    def __init__(self,
                 n_jobs: int,
                 n_succeeded: int,
                 n_failed: int,
                 n_cancelled: int,
                 n_running: int,
                 n_ready: int,
                 n_creating: int):
        self.n_jobs = n_jobs
        self.n_succeeded = n_succeeded
        self.n_failed = n_failed
        self.n_cancelled = n_cancelled
        self.n_running = n_running
        self.n_ready = n_ready
        self.n_creating = n_creating

    def get_value_from_job_state(self, state: 'JobState'):
        if state == JobState.SUCCEEDED:
            return self.n_succeeded
        elif state == JobState.FAILED:
            return self.n_failed
        elif state == JobState.CANCELLED:
            return self.n_cancelled
        elif state == JobState.READY:
            return self.n_ready
        elif state == JobState.CREATING:
            return self.n_creating
        assert state == JobState.RUNNING
        return self.n_running


class MarkJobCompleteColumn(ProgressColumn):
    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:>1.0f} jobs/s", style="progress.data.speed")


class BatchStatusProgress:
    def __init__(self, batch_client, batch_ids: Optional[List[int]], limit: int):
        self.batch_client = batch_client
        self.batch_ids = batch_ids
        self._job_states: Dict[int, Dict[JobState, StateData]] = collections.defaultdict(dict)
        self._batch_tasks: Dict[int, TaskID] = {}
        self._progress = MultiStateProgress(
            "{task.description}",
            MultiStateProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.completed]{task.completed}/{task.total} jobs"),
            MarkJobCompleteColumn(),
            TimeElapsedColumn(),
            SpinnerColumn(style=Style(color=Color.from_rgb(50, 175, 255))),
            max_visible_tasks=limit,
            get_time=lambda: datetime.datetime.utcnow().timestamp()
        )

    async def get_batches(self) -> List[Batch]:
        if self.batch_ids is not None:
            funcs = [functools.partial(self.batch_client.get_batch, batch_id) for batch_id in self.batch_ids]
            return await bounded_gather(*funcs, cancel_on_error=True)
        batches = [b async for b in self.batch_client.list_batches(q='running', limit=10000)]
        batches.sort(key=lambda b: b.id)
        return batches

    async def _initialize_batch(self, batch_id: int, status: dict):
        job_stats = JobStats.from_batch_status(status)
        start_time = datetime.datetime.strptime(status['time_created'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
        deploy_config = get_deploy_config()
        url = deploy_config.external_url('batch', f'/batches/{batch_id}')
        t = self._progress.add_task(f'[link={url}]{batch_id}[/link]', total=job_stats.n_jobs, start_time=start_time)
        self._batch_tasks[batch_id] = t
        for state in JobState:
            value = job_stats.get_value_from_job_state(state)
            state_info = state.value
            state_id = self._progress.add_state(t,
                                                state_info.label,
                                                value,
                                                state_info.style,
                                                JobState.is_complete_state(state))
            self._job_states[batch_id][state] = StateData(t, state_id, value)

    async def initialize(self):
        for b in await self.get_batches():
            await self._initialize_batch(b.id, b._last_known_status)

    async def update(self) -> bool:
        changed = False
        cur_batch_ids = list(self._batch_tasks.keys())
        new_batches = {b.id: b for b in await self.get_batches()}

        batches_to_hide = set(cur_batch_ids).difference(new_batches.keys())
        for batch_id in batches_to_hide:
            task_id = self._batch_tasks[batch_id]
            self._progress.remove_task(task_id, refresh=True)
            del self._batch_tasks[batch_id]

        for batch_id, b in new_batches.items():
            if batch_id not in self._batch_tasks:
                await self._initialize_batch(batch_id, b._last_known_status)
                changed = True
            else:
                t = self._batch_tasks[b.id]
                job_stats = JobStats.from_batch_status(b._last_known_status)
                job_states = self._job_states[b.id]
                for state, state_data in job_states.items():
                    new_value = job_stats.get_value_from_job_state(state)
                    changed |= (state_data.value != new_value)
                    state_data.value = new_value
                    self._progress.update_state(state_data.task_id, state_data.state_id, completed=new_value)
                self._progress.update(t, total=job_stats.n_jobs)
        return changed

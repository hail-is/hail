from collections import defaultdict, namedtuple
from enum import Enum
from typing import Dict
from rich.progress import TextColumn, TaskID
from rich.style import Style

from ..rich_multistate_progress_bar import MultiStateProgressColumn, MultiStateProgress
from .common import StateData


ClusterStyle = namedtuple('ClusterStyle', ['label', 'style'])


class ClusterState(Enum):
    ME = ClusterStyle('Me', Style(color='magenta'))
    OTHER_USERS = ClusterStyle('Other Users', Style(color='cyan'))
    AVAILABLE = ClusterStyle('Available', Style(color='green'))
    PROVISIONING = ClusterStyle('Provisioning', Style(color='yellow'))

    @staticmethod
    def is_complete_state(state: 'ClusterState'):
        return state == ClusterState.ME


class PoolStats:
    @staticmethod
    def from_dict(pool: dict) -> 'PoolStats':
        name = pool['name']
        cores_mcpu_by_state = pool['all_versions_cores_mcpu_by_state']
        total_cores = pool['total_capacity_cores']
        pending_cores = cores_mcpu_by_state.get('pending', 0) / 1000
        active_cores = cores_mcpu_by_state.get('active', 0) / 1000

        me_cores = pool['user_running_cores_mcpu'] / 1000
        provisioning_cores = pending_cores
        available_cores = pool['current_worker_version_active_schedulable_free_cores_mcpu'] / 1000
        other_users_cores = max(0, active_cores - me_cores - available_cores)
        assert 0 <= other_users_cores + me_cores + available_cores + provisioning_cores <= total_cores
        return PoolStats(name, total_cores, me_cores, other_users_cores, available_cores, provisioning_cores)

    def __init__(self, name: str, total_cores, me_cores, other_users_cores, available_cores, provisioning_cores):
        self.name = name
        self.total_cores = total_cores
        self.me_cores = me_cores
        self.other_users_cores = other_users_cores
        self.available_cores = available_cores
        self.provisioning_cores = provisioning_cores

    def get_value_from_cluster_state(self, state: 'ClusterState'):
        if state == ClusterState.ME:
            return self.me_cores
        elif state == ClusterState.OTHER_USERS:
            return self.other_users_cores
        elif state == ClusterState.AVAILABLE:
            return self.available_cores
        assert state == ClusterState.PROVISIONING
        return self.provisioning_cores


class ClusterCapacityProgress:
    def __init__(self, batch_client):
        self.batch_client = batch_client
        self._progress = MultiStateProgress(
            "{task.description}",
            MultiStateProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.total]{task.total} cores"),
            max_visible_tasks=None
        )
        self._pool_states: Dict[str, Dict[ClusterState, StateData]] = defaultdict(dict)
        self._pool_tasks: Dict[str, TaskID] = {}

    async def cluster_stats(self):
        return await self.batch_client.cluster_stats()

    async def initialize(self):
        cluster_stats = await self.cluster_stats()
        for pool in cluster_stats.values():
            self._initialize_pool(pool)

    def _initialize_pool(self, pool: dict):
        pool_stats = PoolStats.from_dict(pool)

        t = self._progress.add_task(pool_stats.name, total=pool_stats.total_cores)
        self._pool_tasks[pool_stats.name] = t

        for state in ClusterState:
            value = pool_stats.get_value_from_cluster_state(state)
            state_info = state.value
            state_id = self._progress.add_state(t,
                                                state_info.label,
                                                value,
                                                state_info.style,
                                                ClusterState.is_complete_state(state))
            self._pool_states[pool_stats.name][state] = StateData(t, state_id, value)

    async def update(self) -> bool:
        changed = False
        cluster_stats = await self.cluster_stats()
        for pool in cluster_stats.values():
            pool_stats = PoolStats.from_dict(pool)
            if pool_stats.name not in self._pool_states:
                self._initialize_pool(pool)
            t = self._pool_tasks[pool_stats.name]
            pool_states = self._pool_states[pool_stats.name]
            for state, state_data in pool_states.items():
                new_value = pool_stats.get_value_from_cluster_state(state)
                changed |= (new_value != state_data.value)
                state_data.value = new_value
                self._progress.update_state(state_data.task_id, state_data.state_id, completed=new_value)
            self._progress.update(t, total=pool_stats.total_cores)
        return changed

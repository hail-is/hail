from typing import Any, Dict, List, Literal, Optional, TypedDict

from typing_extensions import NotRequired


class CostBreakdownEntry(TypedDict):
    resource: str
    cost: float


class GetJobResponseV1Alpha(TypedDict):
    batch_id: int
    job_id: int
    job_group_id: int
    name: Optional[str]
    user: str
    billing_project: str
    state: Literal['Pending', 'Ready', 'Creating', 'Running', 'Failed', 'Cancelled', 'Error', 'Success']
    exit_code: Optional[int]
    duration: Optional[int]
    cost: Optional[float]
    msec_mcpu: int
    cost_breakdown: List[CostBreakdownEntry]
    status: Optional[Dict[str, Any]]
    spec: Optional[Dict[str, Any]]
    attributes: NotRequired[Dict[str, str]]
    always_run: bool
    display_state: Optional[str]


class JobListEntryV1Alpha(TypedDict):
    batch_id: int
    job_id: int
    job_group_id: int
    name: Optional[str]
    user: str
    billing_project: str
    state: Literal['Pending', 'Ready', 'Creating', 'Running', 'Failed', 'Cancelled', 'Error', 'Success']
    exit_code: Optional[int]
    duration: Optional[int]
    cost: Optional[float]
    msec_mcpu: int
    cost_breakdown: List[CostBreakdownEntry]
    always_run: bool
    display_state: Optional[str]


class GetJobsResponseV1Alpha(TypedDict):
    jobs: List[JobListEntryV1Alpha]
    last_job_id: NotRequired[int]


class GetJobGroupResponseV1Alpha(TypedDict):
    batch_id: int
    job_group_id: int
    state: Literal['failure', 'cancelled', 'success', 'running']
    complete: bool
    n_jobs: int
    n_completed: int
    n_succeeded: int
    n_failed: int
    n_cancelled: int
    time_created: Optional[str]  # date string
    time_completed: Optional[str]  # date string
    duration: Optional[int]
    cost: float
    cost_breakdown: List[CostBreakdownEntry]
    attributes: NotRequired[Dict[str, str]]

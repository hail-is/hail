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
    end_time: Optional[str]
    duration: Optional[int]
    cost: Optional[float]
    msec_mcpu: int
    cost_breakdown: List[CostBreakdownEntry]
    status: Optional[Dict[str, Any]]
    spec: Optional[Dict[str, Any]]
    attributes: NotRequired[Dict[str, str]]
    always_run: bool
    n_max_attempts: int
    display_state: Optional[str]
    inst_coll: NotRequired[str]


class JobListEntryV1Alpha(TypedDict):
    batch_id: int
    job_id: int
    job_group_id: int
    name: Optional[str]
    user: str
    billing_project: str
    state: Literal['Pending', 'Ready', 'Creating', 'Running', 'Failed', 'Cancelled', 'Error', 'Success']
    exit_code: Optional[int]
    end_time: Optional[str]
    duration: Optional[int]
    cost: Optional[float]
    msec_mcpu: int
    cost_breakdown: List[CostBreakdownEntry]
    always_run: bool
    n_max_attempts: int
    display_state: Optional[str]


class GetJobsResponseV1Alpha(TypedDict):
    jobs: List[JobListEntryV1Alpha]
    last_job_id: NotRequired[int]


class JobOffsetPagination(TypedDict):
    current_job_offset: int
    next_page_job_offset: Optional[int]
    page_size: int
    total_jobs: int


class AttemptTimingV1Alpha(TypedDict):
    attempt_id: str
    start_time: Optional[int]
    end_time: Optional[int]
    reason: Optional[str]


class JobTimingV1Alpha(TypedDict):
    job_id: int
    attempts: List[AttemptTimingV1Alpha]


class GetBatchTimingResponseV1Alpha(TypedDict):
    data: List[JobTimingV1Alpha]
    pagination: JobOffsetPagination


class JobGraphEntryV1Alpha(TypedDict):
    job_id: int
    parent_ids: List[int]


class GetJobGraphResponseV1Alpha(TypedDict):
    data: List[JobGraphEntryV1Alpha]
    pagination: JobOffsetPagination


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

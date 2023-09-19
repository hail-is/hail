from typing import TypedDict, Literal, Optional, List, Any, Dict
from typing_extensions import NotRequired


class CostBreakdownEntry(TypedDict):
    resource: str
    cost: float


class GetJobResponseV1Alpha(TypedDict):
    batch_id: int
    job_id: int
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


class JobListEntryV1Alpha(TypedDict):
    batch_id: int
    job_id: int
    name: Optional[str]
    user: str
    billing_project: str
    state: Literal['Pending', 'Ready', 'Creating', 'Running', 'Failed', 'Cancelled', 'Error', 'Success']
    exit_code: Optional[int]
    duration: Optional[int]
    cost: Optional[float]
    msec_mcpu: int
    cost_breakdown: List[CostBreakdownEntry]


class GetJobsResponseV1Alpha(TypedDict):
    jobs: List[JobListEntryV1Alpha]
    last_job_id: NotRequired[int]

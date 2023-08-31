from typing import TypedDict, Literal, Optional, List, Any, Dict
from typing_extensions import NotRequired


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


class GetJobsResponseV1Alpha(TypedDict):
    jobs: List[JobListEntryV1Alpha]
    last_job_id: NotRequired[int]

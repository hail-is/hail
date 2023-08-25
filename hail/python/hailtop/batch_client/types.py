from typing import TypedDict, Literal, Optional, List
from typing_extensions import NotRequired


class JobListEntryV1Alpha(TypedDict):
    batch_id: int
    job_id: int
    name: str
    user: str
    billing_project: str
    state: Literal['Pending', 'Ready', 'Creating', 'Running', 'Failed', 'Cancelled', 'Error', 'Success']
    exit_code: Optional[int]
    duration: Optional[int]
    cost: float
    msec_mcpu: int


class GetJobsResponseV1Alpha(TypedDict):
    jobs: List[JobListEntryV1Alpha]
    last_job_id: NotRequired[int]

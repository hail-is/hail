from rich.progress import TaskID


class StateData:
    def __init__(self, task_id: TaskID, state_id: int, value: int):
        self.task_id = task_id
        self.state_id = state_id
        self.value = value

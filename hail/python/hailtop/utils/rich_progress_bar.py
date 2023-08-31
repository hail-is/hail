from typing import Optional, Callable, Tuple
from rich.progress import MofNCompleteColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, Progress, ProgressColumn, TaskProgressColumn


class SimpleRichProgressBarTask:
    def __init__(self, progress: Progress, tid):
        self.progress = progress
        self.tid = tid

    def total(self) -> int:
        assert len(self.progress.tasks) == 1
        return int(self.progress.tasks[0].total or 0)

    def update(self, delta_n: int, *, total: Optional[int] = None):
        self.progress.update(self.tid, advance=delta_n, total=total)

    def make_listener(self) -> Callable[[int], None]:
        return make_listener(self.progress, self.tid)


class SimpleRichProgressBar:
    def __init__(self, *args, description: Optional[str] = None, total: int, visible: bool = True, **kwargs):
        self.description = description
        self.total = total
        self.visible = visible
        if len(args) == 0:
            args = RichProgressBar.get_default_columns()
        self.progress = Progress(*args, **kwargs)

    def __enter__(self) -> SimpleRichProgressBarTask:
        self.progress.start()
        tid = self.progress.add_task(self.description or '', total=self.total, visible=self.visible)
        return SimpleRichProgressBarTask(self.progress, tid)

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type
        del exc_value
        del traceback
        try:
            self.progress.refresh()
        finally:
            self.progress.stop()


def make_listener(progress: Progress, tid) -> Callable[[int], None]:
    total = 0

    def listen(delta: int):
        nonlocal total
        if delta > 0:
            total += delta
            progress.update(tid, total=total)
        else:
            progress.update(tid, advance=-delta)
    return listen


class RichProgressBar:
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            args = RichProgressBar.get_default_columns()
        self.progress = Progress(*args, **kwargs)

    @staticmethod
    def get_default_columns() -> Tuple[ProgressColumn, ...]:
        return (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="bar.finished"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn()
        )

    def __enter__(self) -> Progress:
        self.progress.start()
        return self.progress

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type
        del exc_value
        del traceback
        try:
            self.progress.refresh()
        finally:
            self.progress.stop()


class BatchProgressBar:
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            args = BatchProgressBar.get_default_columns()
        self.progress = Progress(*args, **kwargs)

    @staticmethod
    def get_default_columns() -> Tuple[ProgressColumn, ...]:
        return (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="bar.finished"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn()
        )

    def __enter__(self) -> 'BatchProgressBar':
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type
        del exc_value
        del traceback
        try:
            self.progress.refresh()
        finally:
            self.progress.stop()

    def with_task(self, description: str, *, total: int = 0, disable: bool = False, transient: bool = False) -> 'BatchProgressBarTask':
        tid = self.progress.add_task(description, total=total, disable=disable)
        return BatchProgressBarTask(self.progress, tid, transient)


class BatchProgressBarTask:
    def __init__(self, progress: Progress, tid, transient: bool):
        self.progress = progress
        self.tid = tid
        self.transient = transient

    def total(self) -> int:
        assert len(self.progress.tasks) == 1
        return int(self.progress.tasks[0].total or 0)

    def __enter__(self) -> 'BatchProgressBarTask':
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type
        del exc_value
        del traceback
        if self.transient:
            self.progress.remove_task(self.tid)

    def update(self, advance: Optional[int] = None, **kwargs):
        self.progress.update(self.tid, advance=advance, **kwargs)


def is_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython  # pylint: disable=import-error,import-outside-toplevel
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except (NameError, ModuleNotFoundError):
        return False

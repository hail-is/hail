import collections
import math
from functools import lru_cache
from time import monotonic
from typing import Any, Union, Dict, Iterable, List, Optional
from threading import RLock

from rich.progress import (
    RenderableType,
    Group,
    GetTimeCallable,
    Progress,
    ProgressSample,
    TextColumn,
    ProgressColumn,
    Column,
    Task,
    TaskID,
)
from rich.segment import Segments
from rich.table import Table
from rich.text import Text
from rich.color import Color, blend_rgb
from rich.color_triplet import ColorTriplet
from rich.console import Console, ConsoleOptions, RenderResult
from rich.jupyter import JupyterMixin
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style, StyleType

# Number of characters before 'pulse' animation repeats
PULSE_SIZE = 20


class ProgressBarState:
    def __init__(self, description: str, val: float, style: StyleType, is_complete: bool):
        self.description = description
        self.val = val
        self.style = style
        self.is_complete = is_complete


class MultiStateProgressBar(JupyterMixin):
    def __init__(
        self,
        total: Optional[float] = 100.0,
        states: Optional[List[ProgressBarState]] = None,
        width: Optional[int] = None,
        pulse: bool = False,
        style: StyleType = "bar.back",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        animation_time: Optional[float] = None,
    ):
        self.total = total
        self.states = states or []
        self.width = width
        self.pulse = pulse
        self.style = style
        self.finished_style = finished_style
        self.pulse_style = pulse_style
        self.animation_time = animation_time

        self._pulse_segments: Optional[List[Segment]] = None

    def __repr__(self) -> str:
        return f"<Bar {self.completed!r} of {self.total!r}>"

    @property
    def completed(self):
        if self.states:
            return sum(s.val for s in self.states if s.is_complete)
        return 0

    @property
    def percentage_completed(self) -> Optional[float]:
        """Calculate percentage complete."""
        if self.total is None:
            return None
        completed = (self.completed / self.total) * 100.0
        completed = min(100, max(0.0, completed))
        return completed

    @lru_cache(maxsize=16)
    def _get_pulse_segments(
        self,
        fore_style: Style,
        back_style: Style,
        color_system: str,
        no_color: bool,
        ascii: bool = False,
    ) -> List[Segment]:
        """Get a list of segments to render a pulse animation.

        Returns:
            List[Segment]: A list of segments, one segment per character.
        """
        bar = "-" if ascii else "━"
        segments: List[Segment] = []
        if color_system not in ("standard", "eight_bit", "truecolor") or no_color:
            segments += [Segment(bar, fore_style)] * (PULSE_SIZE // 2)
            segments += [Segment(" " if no_color else bar, back_style)] * (PULSE_SIZE - (PULSE_SIZE // 2))
            return segments

        append = segments.append
        fore_color = fore_style.color.get_truecolor() if fore_style.color else ColorTriplet(255, 0, 255)
        back_color = back_style.color.get_truecolor() if back_style.color else ColorTriplet(0, 0, 0)
        cos = math.cos
        pi = math.pi
        _Segment = Segment
        _Style = Style
        from_triplet = Color.from_triplet

        for index in range(PULSE_SIZE):
            position = index / PULSE_SIZE
            fade = 0.5 + cos((position * pi * 2)) / 2.0
            color = blend_rgb(fore_color, back_color, cross_fade=fade)
            append(_Segment(bar, _Style(color=from_triplet(color))))
        return segments

    def update(self, total: Optional[float] = None) -> None:
        """Update progress with new values.

        Args:
            completed (float): Number of steps completed.
            total (float, optional): Total number of steps, or ``None`` to not change. Defaults to None.
        """
        self.total = total if total is not None else self.total

    def _render_pulse(self, console: Console, width: int, ascii: bool = False) -> Iterable[Segment]:
        """Renders the pulse animation.

        Args:
            console (Console): Console instance.
            width (int): Width in characters of pulse animation.

        Returns:
            RenderResult: [description]

        Yields:
            Iterator[Segment]: Segments to render pulse
        """
        fore_style = console.get_style(self.pulse_style, default="white")
        back_style = console.get_style(self.style, default="black")

        pulse_segments = self._get_pulse_segments(
            fore_style, back_style, console.color_system, console.no_color, ascii=ascii
        )
        segment_count = len(pulse_segments)
        current_time = monotonic() if self.animation_time is None else self.animation_time
        segments = pulse_segments * (int(width / segment_count) + 2)
        offset = int(-current_time * 15) % segment_count
        segments = segments[offset : offset + width]
        yield from segments

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:

        width = min(self.width or options.max_width, options.max_width)
        ascii = options.legacy_windows or options.ascii_only
        should_pulse = self.pulse or self.total is None
        if should_pulse:
            yield from self._render_pulse(console, width, ascii=ascii)
            return

        segments = []
        total_bar_count = 0
        total_half_bar_count = 0

        bar = "-" if ascii else "█"  # "━"
        half_bar_right = " " if ascii else "█"  # "╸"
        half_bar_left = " " if ascii else "█"  # "╺"

        _Segment = Segment

        for state in self.states:
            completed = min(self.total, max(0, state.val)) if self.total is not None else None

            complete_halves = (
                int(width * 2 * completed / self.total) if self.total and completed is not None else width * 2
            )
            bar_count = complete_halves // 2
            total_bar_count += bar_count
            half_bar_count = complete_halves % 2
            total_half_bar_count += half_bar_count

            if bar_count:
                segments.append(_Segment(bar * bar_count, state.style))
            if half_bar_count:
                segments.append(_Segment(half_bar_right * half_bar_count, state.style))

        style = console.get_style(self.style)

        if not console.no_color:
            remaining_bars = width - total_bar_count - total_half_bar_count
            if remaining_bars and console.color_system is not None:
                if not total_half_bar_count and total_bar_count:
                    segments.append(_Segment(half_bar_left, style))
                    remaining_bars -= 1
                if remaining_bars:
                    segments.append(_Segment(bar * remaining_bars, style))

        assert segments

        yield Segments(segments)

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(self.width, self.width) if self.width is not None else Measurement(4, options.max_width)


class MultiStateTask(Task):
    def __init__(
        self,
        id: TaskID,
        description: str,
        total: Optional[float],
        _get_time: GetTimeCallable,
        finished_time: Optional[float] = None,
        visible: bool = True,
        fields: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        stop_time: Optional[float] = None,
        finished_speed: Optional[float] = None,
        _lock: Optional[RLock] = None,
    ):
        self.id = id
        self.description = description
        self.total = total
        self._get_time = _get_time
        self.finished_time = finished_time
        self.visible = visible
        self.fields = fields or {}
        self.start_time = start_time
        self.stop_time = stop_time
        self.finished_speed = finished_speed
        self.states: List[ProgressBarState] = []
        self._progress = collections.deque(maxlen=1000)
        self._lock = _lock or RLock()
        self._old_sample_time = None
        self._last_completed = 0

    @property
    def completed(self):
        if self.states:
            return sum(s.val for s in self.states if s.is_complete)
        return None

    @property
    def percentage(self) -> Optional[float]:
        """Calculate percentage complete."""
        if self.total is None:
            return None
        completed = (self.completed / self.total) * 100.0
        completed = min(100, max(0.0, completed))
        return completed

    def add_state(self, state: ProgressBarState) -> int:
        self.states.append(state)
        return len(self.states) - 1

    @property
    def speed(self) -> Optional[float]:
        """Optional[float]: Get the estimated speed in steps per second."""
        if self.start_time is None:
            return None
        with self._lock:
            progress = self._progress
            if not progress:
                return None
            total_time = progress[-1].timestamp - progress[0].timestamp
            if total_time == 0:
                return None
            iter_progress = iter(progress)
            next(iter_progress)
            total_completed = sum(sample.completed for sample in iter_progress)
            speed = total_completed / total_time
            return speed

    @property
    def time_remaining(self) -> Optional[float]:
        """Optional[float]: Get estimated time to completion, or ``None`` if no data."""
        if self.finished:
            return 0.0
        speed = self.speed
        if not speed:
            return None
        remaining = self.remaining
        if remaining is None:
            return None
        estimate = math.ceil(remaining / speed)
        return estimate

    def _reset(self) -> None:
        """Reset progress."""
        self._progress.clear()
        self.finished_time = None
        self.finished_speed = None
        self._last_completed = 0


class BatchMarkJobCompleteColumn(ProgressColumn):
    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:>1.0f} jobs/s", style="progress.data.speed")


class MultiStateProgressColumn(ProgressColumn):
    """Renders a visual progress bar.

    Args:
        bar_width (Optional[int], optional): Width of bar or None for full width. Defaults to 40.
        style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
        complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
        finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
        pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
    """

    def __init__(
        self,
        bar_width: Optional[int] = 40,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        pulse_style: StyleType = "bar.pulse",
        table_column: Optional[Column] = None,
    ) -> None:
        self.bar_width = bar_width
        self.style = style
        self.complete_style = complete_style
        self.pulse_style = pulse_style
        super().__init__(table_column=table_column)

    def render(self, task: "MultiStateTask") -> MultiStateProgressBar:
        """Gets a progress bar widget for a task."""
        return MultiStateProgressBar(
            total=max(0, task.total) if task.total is not None else None,
            states=task.states,
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=False,
            animation_time=task.get_time(),
            style=self.style,
            pulse_style=self.pulse_style,
        )


class MultiStateProgress(Progress):
    _tasks: Dict[TaskID, MultiStateTask] = {}

    def __init__(
        self,
        *columns: Union[str, ProgressColumn],
        console: Optional[Console] = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 10,
        speed_estimate_period: float = 30.0,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: Optional[GetTimeCallable] = None,
        disable: bool = False,
        expand: bool = False,
        max_visible_tasks: Optional[int] = 5,
        show_head: bool = True,
    ):
        self._max_n_visible_tasks = max_visible_tasks
        self._show_head = show_head
        super().__init__(*columns,
                         console=console,
                         auto_refresh=auto_refresh,
                         refresh_per_second=refresh_per_second,
                         speed_estimate_period=speed_estimate_period,
                         transient=transient,
                         redirect_stdout=redirect_stdout,
                         redirect_stderr=redirect_stderr,
                         get_time=get_time,
                         disable=disable,
                         expand=expand)

    def update_state(
        self,
        task_id: TaskID,
        state_id: int,
        *,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        refresh: bool = False,
    ):
        with self._lock:
            task = self._tasks[task_id]
            state = task.states[state_id]

            if advance is not None:
                state.val += advance
            if completed is not None:
                state.val = completed

        if refresh:
            self.refresh()

    def _update_visible_tasks(self):
        tasks = list(self._tasks.values())

        if self._show_head:
            for idx, t in enumerate(tasks):
                visible = self._max_n_visible_tasks is None or idx < self._max_n_visible_tasks
                t.visible = visible
        else:
            for idx, t in enumerate(tasks):
                visible = self._max_n_visible_tasks is None or len(tasks) - idx < self._max_n_visible_tasks
                t.visible = visible

    def remove_task(self, task_id: TaskID, refresh: bool = False) -> None:
        """Delete a task if it exists.

        Args:
            task_id (TaskID): A task ID.

        """
        with self._lock:
            del self._tasks[task_id]

        self._update_visible_tasks()

        if refresh:
            self.refresh()

    def update(
        self,
        task_id: TaskID,
        *,
        total: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        """Update information associated with a task.

        Args:
            task_id (TaskID): Task id (returned by add_task).
            total (float, optional): Updates task.total if not None.
            description (str, optional): Change task description if not None.
            visible (bool, optional): Set visible flag if not None.
            refresh (bool): Force a refresh of progress information. Default is False.
            **fields (Any): Additional data fields required for rendering.
        """
        with self._lock:
            task = self._tasks[task_id]

            if total is not None and total != task.total:
                task.total = total
                task._reset()
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)

            new_completed = task.completed
            update_completed = new_completed - task._last_completed

            current_time = self.get_time()
            _progress = task._progress

            popleft = _progress.popleft
            while _progress and task._old_sample_time is not None and _progress[0].timestamp < task._old_sample_time:
                popleft()
            if update_completed > 0:
                _progress.append(ProgressSample(current_time, update_completed))
            if task.total is not None and new_completed >= task.total and task.finished_time is None:
                task.finished_time = task.elapsed

            task._old_sample_time = current_time
            task._last_completed = new_completed

        if refresh:
            self.refresh()

    def add_state(
        self,
        task_id: TaskID,
        description: str,
        value: float,
        style: StyleType,
        is_complete: bool
    ) -> int:
        state = ProgressBarState(description, value, style, is_complete)
        task = self._tasks[task_id]
        state_id = task.add_state(state)
        self.update(task_id, refresh=False)
        return state_id

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: Optional[float] = 100.0,
        visible: bool = True,
        refresh: bool = False,
        start_time: Optional[float] = None,
        **fields: Any,
    ) -> TaskID:
        """Add a new 'task' to the Progress display.

        Args:
            description (str): A description of the task.
            start (bool, optional): Start the task immediately (to calculate elapsed time). If set to False,
                you will need to call `start` manually. Defaults to True.
            total (float, optional): Number of total steps in the progress if known.
                Set to None to render a pulsing animation. Defaults to 100.
            completed (int, optional): Number of steps completed so far. Defaults to 0.
            visible (bool, optional): Enable display of the task. Defaults to True.
            **fields (str): Additional data fields required for rendering.

        Returns:
            TaskID: An ID you can use when calling `update`.
        """
        with self._lock:
            task = MultiStateTask(
                self._task_index,
                description,
                total,
                start_time=start_time,
                visible=visible,
                fields=fields,
                _get_time=self.get_time,
                _lock=self._lock,
            )
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)

            self._update_visible_tasks()

            new_task_index = self._task_index
            self._task_index = TaskID(int(self._task_index) + 1)

        if refresh:
            self.refresh()

        return new_task_index

    def make_legend(self) -> Optional[Table]:
        if not self._tasks:
            return None
        tasks = list(self._tasks.values())
        current_task = tasks[-1]
        columns = [TextColumn('legend: ')]
        all_states = sorted([state for task in tasks for state in task.states], key=lambda x: x.description)

        seen = set()
        for state in all_states:
            description = state.description
            if description not in seen:
                columns.append(TextColumn(f"━ {description}", state.style))
                seen.add(description)

        table_columns = (
            (Column(no_wrap=True) if isinstance(_column, str) else _column.get_table_column().copy())
            for _column in columns
        )
        table = Table.grid(*table_columns, padding=(0, 2), expand=self.expand)

        table.add_row(
            *(
                (column.format(task=current_task) if isinstance(column, str) else column(current_task))
                for column in columns
            )
        )

        return table

    def make_previous_tasks_table(self) -> Optional[Table]:
        tasks = list(self._tasks.values())
        current_task = tasks[0]  # dummy
        previous_n_tasks = len(tasks) - self._max_n_visible_tasks
        assert previous_n_tasks > 0
        columns = [TextColumn(f'{previous_n_tasks} task(s) previously completed')]

        table_columns = (
            (Column(no_wrap=True) if isinstance(_column, str) else _column.get_table_column().copy())
            for _column in columns
        )
        table = Table.grid(*table_columns, padding=(0, 2), expand=self.expand)

        table.add_row(
            *(
                (column.format(task=current_task) if isinstance(column, str) else column(current_task))
                for column in columns
            )
        )

        return table

    def make_overflow_tasks_table(self) -> Optional[Table]:
        tasks = list(self._tasks.values())
        current_task = tasks[0]  # dummy
        previous_n_tasks = len(tasks) - self._max_n_visible_tasks
        assert previous_n_tasks > 0
        columns = [TextColumn(f'{previous_n_tasks} items hidden')]

        table_columns = (
            (Column(no_wrap=True) if isinstance(_column, str) else _column.get_table_column().copy())
            for _column in columns
        )
        table = Table.grid(*table_columns, padding=(0, 2), expand=self.expand)

        table.add_row(
            *(
                (column.format(task=current_task) if isinstance(column, str) else column(current_task))
                for column in columns
            )
        )

        return table

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        """Get a table to render the Progress display.

        Args:
            tasks (Iterable[Task]): An iterable of Task instances, one per row of the table.

        Returns:
            Table: A table instance.
        """
        n_visible_tasks = len([t for t in tasks if t.visible])
        if n_visible_tasks == 0:
            table = Table.grid(padding=(0, 1), expand=self.expand)
            table.add_row('No items to display.')
            return table

        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )
        table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)

        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        (
                            column.format(task=task)
                            if isinstance(column, str)
                            else column(task)
                        )
                        for column in self.columns
                    )
                )
        return table

    def get_renderable(self) -> RenderableType:
        """Get a renderable for the progress display."""
        renderable = Group(*self.get_renderables())
        return renderable

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        if not self._show_head and self._max_n_visible_tasks and len(self.tasks) > self._max_n_visible_tasks:
            yield self.make_previous_tasks_table()
            empty_table = Table.grid(padding=(0, 2), expand=self.expand)
            yield empty_table

        main_table = self.make_tasks_table(self.tasks)
        yield main_table
        empty_table = Table.grid(padding=(0, 2), expand=self.expand)
        yield empty_table

        if self._show_head and self._max_n_visible_tasks and len(self.tasks) > self._max_n_visible_tasks:
            yield self.make_overflow_tasks_table()
            empty_table = Table.grid(padding=(0, 2), expand=self.expand)
            yield empty_table
            
        legend_table = self.make_legend()
        if legend_table:
            yield legend_table

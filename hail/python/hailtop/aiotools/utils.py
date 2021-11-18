from typing import Callable


def make_tqdm_listener(pbar) -> Callable[[int], None]:
    def listener(delta):
        if pbar.total is None:
            pbar.total = 0
        if delta > 0:
            pbar.total += delta
            pbar.refresh()
        if delta < 0:
            pbar.update(-delta)
    return listener

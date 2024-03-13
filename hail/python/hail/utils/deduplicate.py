from typing import Iterable, List, Optional, Tuple


def deduplicate(
    ids: Iterable[str], *, max_attempts: Optional[int] = None, already_used: Optional[Iterable[str]] = None
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Deduplicate the strings in `ids`.

    Example
    -------

    >>> deduplicate(['a', 'a', 'a'])
    ([('a', 'a_1'), ('a', 'a_2')], ['a', 'a_1', 'a_2'])

    >>> deduplicate(['a', 'a_1', 'a'])
    ([('a', 'a_2')], ['a', 'a_1', 'a_2'])

    Parameters
    ----------
    ids : list of :class:`str`
        The list of strings, possibly containing duplicates.

    Returns
    -------
    list of pairs of :obj:`.str` and :obj:`.str`
        A string and the deduplicated string, for each occurrence after the
        first.

    list of :class:`str`
        A list, equal in length to `ids`, without duplicates.
    """
    uniques = set(already_used) if already_used else set()
    mapping = []
    new_ids = []

    def fmt(s, i):
        return '{}_{}'.format(s, i)

    for s in ids:
        s_ = s
        i = 0
        while s_ in uniques:
            i += 1
            if max_attempts and i > max_attempts:
                raise RecursionError(f'cannot deduplicate {s} after {max_attempts} attempts')
            s_ = fmt(s, i)

        if s_ != s:
            mapping.append((s, s_))
        uniques.add(s_)
        new_ids.append(s_)

    return mapping, new_ids

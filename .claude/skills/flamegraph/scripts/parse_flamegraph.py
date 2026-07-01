#!/usr/bin/env python3
"""Parse async-profiler flamegraph HTML files and extract sample data.

Usage:
    # Single file summary
    python parse_flamegraph.py profile.html

    # Compare two profiles
    python parse_flamegraph.py before.html after.html

    # Filter to specific patterns
    python parse_flamegraph.py profile.html --filter 'is/hail'

    # Trace call paths to a hot function
    python parse_flamegraph.py profile.html --callers 'ArrayBuilder'
    python parse_flamegraph.py profile.html --callers 'DECODE' --depth 12

    # Show top N functions
    python parse_flamegraph.py profile.html --top 30

    # JSON output for programmatic use
    python parse_flamegraph.py profile.html --json
"""

import argparse
import json
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Frame:
    name: str
    level: int
    left: int
    width: int


@dataclass
class FlameGraph:
    path: str
    total_samples: int
    frames: list[Frame]
    inclusive: dict[str, int] = field(default_factory=dict)
    self_time: dict[str, int] = field(default_factory=dict)


def parse_flamegraph(filepath: str) -> FlameGraph:
    content = Path(filepath).read_text()

    cpool = _extract_cpool(content)
    frames = _extract_frames(content, cpool)

    if not frames:
        print(f"error: no frames parsed from {filepath}", file=sys.stderr)
        sys.exit(1)

    total = frames[0].width

    inclusive = defaultdict(int)
    for f in frames:
        inclusive[f.name] += f.width

    self_time = _compute_self_time(frames)

    return FlameGraph(
        path=filepath,
        total_samples=total,
        frames=frames,
        inclusive=dict(inclusive),
        self_time=dict(self_time),
    )


def _extract_cpool(content: str) -> list[str]:
    m = re.search(r'const cpool = \[(.*?)\];\s*unpack\(cpool\)', content, re.DOTALL)
    if not m:
        print("error: could not find cpool in file", file=sys.stderr)
        sys.exit(1)

    entries = re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1))
    cpool = list(entries)
    for i in range(1, len(cpool)):
        prefix_len = ord(cpool[i][0]) - 32
        cpool[i] = cpool[i - 1][:prefix_len] + cpool[i][1:]
    return cpool


def _extract_frames(content: str, cpool: list[str]) -> list[Frame]:
    data_start = content.index('unpack(cpool);') + len('unpack(cpool);')
    data_end = content.index('search();', data_start)
    data_section = content[data_start:data_end]

    calls = re.findall(r'(?:^|\n)\s*([fun])\(([0-9, ]+)\)', data_section)

    level0 = 0
    left0 = 0
    width0 = 0
    frames = []

    for call_type, args_str in calls:
        parts = [int(x.strip()) for x in args_str.strip().split(',') if x.strip()]

        if call_type == 'f':
            key, level, left = parts[0], parts[1], parts[2]
            width = parts[3] if len(parts) > 3 else 0
            level0 = level
            left0 += left
            if width:
                width0 = width
            frames.append(Frame(cpool[key >> 3], level0, left0, width0))

        elif call_type == 'u':
            key = parts[0]
            width = parts[1] if len(parts) > 1 else 0
            level0 += 1
            if width:
                width0 = width
            frames.append(Frame(cpool[key >> 3], level0, left0, width0))

        elif call_type == 'n':
            key = parts[0]
            width = parts[1] if len(parts) > 1 else 0
            left0 += width0
            if width:
                width0 = width
            frames.append(Frame(cpool[key >> 3], level0, left0, width0))

    return frames


def _compute_self_time(frames: list[Frame]) -> dict[str, int]:
    """Self-time: a frame's width minus the sum of its direct children."""
    by_level: dict[int, list[Frame]] = defaultdict(list)
    for f in frames:
        by_level[f.level].append(f)
    for lvl in by_level:
        by_level[lvl].sort(key=lambda f: f.left)
    level_lefts: dict[int, list[int]] = {
        lvl: [f.left for f in fs] for lvl, fs in by_level.items()
    }

    children_width: dict[int, int] = defaultdict(int)
    for f in frames:
        parent_lvl = f.level - 1
        if parent_lvl not in level_lefts:
            continue
        lefts = level_lefts[parent_lvl]
        idx = bisect_right(lefts, f.left) - 1
        if idx < 0:
            continue
        parent = by_level[parent_lvl][idx]
        if f.left < parent.left + parent.width:
            children_width[id(parent)] += f.width

    self_time = defaultdict(int)
    for f in frames:
        child_w = children_width.get(id(f), 0)
        st = max(0, f.width - child_w)
        if st > 0:
            self_time[f.name] += st

    return dict(self_time)


def _collapse_recursion(stack: tuple[str, ...]) -> tuple:
    """Collapse consecutive repeated cycles into (cycle_tuple, count) markers.

    Detects repeating subsequences of length 1-3 that occur 3+ times
    and replaces them with a single copy plus a count.  This merges
    stacks that differ only in recursion depth.
    """
    result: list = []
    i = 0
    n = len(stack)
    while i < n:
        matched = False
        for cycle_len in range(1, min(4, (n - i) // 2 + 1)):
            cycle = stack[i : i + cycle_len]
            count = 1
            j = i + cycle_len
            while j + cycle_len <= n and stack[j : j + cycle_len] == cycle:
                count += 1
                j += cycle_len
            if count >= 3:
                result.append((tuple(cycle), count))
                i = j
                matched = True
                break
        if not matched:
            result.append(stack[i])
            i += 1
    return tuple(result)


def _trace_callers(
    fg: FlameGraph, pattern: str, depth: int
) -> list[tuple[tuple, int]]:
    """Find call stacks leading to frames matching pattern, grouped by unique stack.

    Stacks are collapsed so that recursive cycles (A -> B -> A -> B ...)
    appear once with a repetition count, merging stacks that differ only
    in recursion depth.
    """
    pat = re.compile(pattern)

    by_level: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for f in fg.frames:
        by_level[f.level].append((f.left, f.width, f.name))
    for lvl in by_level:
        by_level[lvl].sort()

    targets = [f for f in fg.frames if pat.search(f.name)]

    stacks: dict[tuple, int] = defaultdict(int)
    for f in targets:
        stack = []
        for lvl in range(f.level, -1, -1):
            entries = by_level[lvl]
            lefts = [e[0] for e in entries]
            idx = bisect_right(lefts, f.left) - 1
            if idx >= 0:
                pleft, pwidth, pname = entries[idx]
                if f.left >= pleft and f.left < pleft + pwidth:
                    stack.append(pname)
        stack.reverse()
        collapsed = _collapse_recursion(tuple(stack))
        if depth and len(collapsed) > depth:
            collapsed = collapsed[-depth:]
        stacks[collapsed] += f.width

    return sorted(stacks.items(), key=lambda x: -x[1])


def _format_collapsed_stack(stack: tuple) -> list[str]:
    """Render a collapsed stack as display lines."""
    lines = []
    for element in stack:
        if isinstance(element, tuple):
            cycle, count = element
            for frame in cycle[:-1]:
                lines.append(f"    {frame}")
            lines.append(f"    {cycle[-1]}  x{count}")
        else:
            lines.append(f"    {element}")
    return lines


def _collapsed_stack_to_json(stack: tuple) -> list:
    """Convert a collapsed stack to a JSON-serializable list."""
    frames = []
    for element in stack:
        if isinstance(element, tuple):
            cycle, count = element
            frames.append({"cycle": list(cycle), "count": count})
        else:
            frames.append(element)
    return frames


def print_callers(fg: FlameGraph, pattern: str, top_n: int, depth: int):
    results = _trace_callers(fg, pattern, depth)
    print(f"File: {fg.path}")
    print(f"Total samples: {fg.total_samples}")
    print(f"Pattern: {pattern}")
    print(f"Matching stacks: {len(results)}")
    print()

    print(f"=== Top {top_n} call paths to '{pattern}' (by samples) ===")
    for stack, samples in results[:top_n]:
        pct = samples / fg.total_samples * 100
        print(f"\n  [{samples} samples, {pct:.1f}%]")
        for line in _format_collapsed_stack(stack):
            print(line)
    print()


def print_json_callers(fg: FlameGraph, pattern: str, top_n: int, depth: int):
    results = _trace_callers(fg, pattern, depth)
    out = {
        "file": fg.path,
        "total_samples": fg.total_samples,
        "pattern": pattern,
        "stacks": [
            {
                "samples": samples,
                "pct": round(samples / fg.total_samples * 100, 2),
                "frames": _collapsed_stack_to_json(stack),
            }
            for stack, samples in results[:top_n]
        ],
    }
    print(json.dumps(out, indent=2))


def format_table(rows: list[tuple], headers: tuple, max_name_width: int = 90) -> str:
    lines = []
    fmt = f"{{:<{max_name_width}}} " + " ".join(f"{{:>{w}}}" for w in [8] * (len(headers) - 1))
    lines.append(fmt.format(*headers))
    lines.append("-" * (max_name_width + 9 * (len(headers) - 1)))
    for row in rows:
        name = row[0][:max_name_width]
        lines.append(fmt.format(name, *row[1:]))
    return "\n".join(lines)


def print_single(fg: FlameGraph, top_n: int, filter_pat: Optional[str]):
    print(f"File: {fg.path}")
    print(f"Total samples: {fg.total_samples}")
    print()

    pat = re.compile(filter_pat) if filter_pat else None

    print(f"=== Top {top_n} by inclusive time ===")
    items = sorted(fg.inclusive.items(), key=lambda x: -x[1])
    if pat:
        items = [(n, v) for n, v in items if pat.search(n)]
    rows = []
    for name, count in items[:top_n]:
        pct = f"{count / fg.total_samples * 100:.1f}%"
        rows.append((name, pct, str(count)))
    print(format_table(rows, ("Function", "Pct", "Samples")))

    print()
    print(f"=== Top {top_n} by self time ===")
    items = sorted(fg.self_time.items(), key=lambda x: -x[1])
    if pat:
        items = [(n, v) for n, v in items if pat.search(n)]
    rows = []
    for name, count in items[:top_n]:
        pct = f"{count / fg.total_samples * 100:.1f}%"
        rows.append((name, pct, str(count)))
    print(format_table(rows, ("Function", "Pct", "Samples")))


def print_comparison(fg_a: FlameGraph, fg_b: FlameGraph, top_n: int, filter_pat: Optional[str]):
    pat = re.compile(filter_pat) if filter_pat else None
    label_a = Path(fg_a.path).stem
    label_b = Path(fg_b.path).stem

    print(f"=== Comparison: {label_a} vs {label_b} ===")
    print(f"  {label_a}: {fg_a.total_samples} samples")
    print(f"  {label_b}: {fg_b.total_samples} samples")
    ratio = fg_b.total_samples / fg_a.total_samples if fg_a.total_samples else float('inf')
    print(f"  Ratio ({label_b}/{label_a}): {ratio:.2f}x")
    print()

    all_funcs = set(fg_a.inclusive) | set(fg_b.inclusive)
    if pat:
        all_funcs = {f for f in all_funcs if pat.search(f)}

    diffs = []
    for func in all_funcs:
        pct_a = fg_a.inclusive.get(func, 0) / fg_a.total_samples * 100
        pct_b = fg_b.inclusive.get(func, 0) / fg_b.total_samples * 100
        diff = pct_b - pct_a
        abs_a = fg_a.inclusive.get(func, 0)
        abs_b = fg_b.inclusive.get(func, 0)
        diffs.append((func, pct_a, pct_b, diff, abs_a, abs_b))

    print(f"=== Top {top_n} functions by INCREASED % ({label_b} vs {label_a}, inclusive) ===")
    by_increase = sorted(diffs, key=lambda x: -x[3])
    rows = []
    for func, pa, pb, diff, aa, ab in by_increase[:top_n]:
        if diff > 0.1:
            rows.append((func, f"{pa:.1f}%", f"{pb:.1f}%", f"{diff:+.1f}%", str(aa), str(ab)))
    if rows:
        print(format_table(rows, ("Function", label_a, label_b, "Diff", f"#{label_a}", f"#{label_b}")))
    else:
        print("  (none)")

    print()
    print(f"=== Top {top_n} functions by DECREASED % ({label_b} vs {label_a}, inclusive) ===")
    by_decrease = sorted(diffs, key=lambda x: x[3])
    rows = []
    for func, pa, pb, diff, aa, ab in by_decrease[:top_n]:
        if diff < -0.1:
            rows.append((func, f"{pa:.1f}%", f"{pb:.1f}%", f"{diff:+.1f}%", str(aa), str(ab)))
    if rows:
        print(format_table(rows, ("Function", label_a, label_b, "Diff", f"#{label_a}", f"#{label_b}")))
    else:
        print("  (none)")

    # Self-time comparison
    all_self = set(fg_a.self_time) | set(fg_b.self_time)
    if pat:
        all_self = {f for f in all_self if pat.search(f)}

    self_diffs = []
    for func in all_self:
        pct_a = fg_a.self_time.get(func, 0) / fg_a.total_samples * 100
        pct_b = fg_b.self_time.get(func, 0) / fg_b.total_samples * 100
        diff = pct_b - pct_a
        self_diffs.append((func, pct_a, pct_b, diff))

    print()
    print(f"=== Top {top_n} by self-time change ===")
    by_self_change = sorted(self_diffs, key=lambda x: -abs(x[3]))
    rows = []
    for func, pa, pb, diff in by_self_change[:top_n]:
        if abs(diff) > 0.1:
            rows.append((func, f"{pa:.1f}%", f"{pb:.1f}%", f"{diff:+.1f}%"))
    if rows:
        print(format_table(rows, ("Function", label_a, label_b, "Diff")))
    else:
        print("  (none)")

    # Unique functions
    only_a = {f for f in fg_a.inclusive if f not in fg_b.inclusive}
    only_b = {f for f in fg_b.inclusive if f not in fg_a.inclusive}
    if pat:
        only_a = {f for f in only_a if pat.search(f)}
        only_b = {f for f in only_b if pat.search(f)}

    if only_b:
        print()
        print(f"=== Functions only in {label_b} (by inclusive %) ===")
        items = sorted(only_b, key=lambda f: -fg_b.inclusive[f])
        rows = []
        for func in items[:top_n]:
            pct = f"{fg_b.inclusive[func] / fg_b.total_samples * 100:.1f}%"
            rows.append((func, pct, str(fg_b.inclusive[func])))
        print(format_table(rows, ("Function", "Pct", "Samples")))

    if only_a:
        print()
        print(f"=== Functions only in {label_a} (by inclusive %) ===")
        items = sorted(only_a, key=lambda f: -fg_a.inclusive[f])
        rows = []
        for func in items[:top_n]:
            pct = f"{fg_a.inclusive[func] / fg_a.total_samples * 100:.1f}%"
            rows.append((func, pct, str(fg_a.inclusive[func])))
        print(format_table(rows, ("Function", "Pct", "Samples")))


def print_json_single(fg: FlameGraph, top_n: int, filter_pat: Optional[str]):
    pat = re.compile(filter_pat) if filter_pat else None

    def filtered(d):
        if not pat:
            return d
        return {k: v for k, v in d.items() if pat.search(k)}

    inc = filtered(fg.inclusive)
    st = filtered(fg.self_time)

    result = {
        "file": fg.path,
        "total_samples": fg.total_samples,
        "inclusive_top": sorted(
            [{"name": k, "samples": v, "pct": round(v / fg.total_samples * 100, 2)} for k, v in inc.items()],
            key=lambda x: -x["samples"],  # type: ignore[operator]
        )[:top_n],
        "self_time_top": sorted(
            [{"name": k, "samples": v, "pct": round(v / fg.total_samples * 100, 2)} for k, v in st.items()],
            key=lambda x: -x["samples"],  # type: ignore[operator]
        )[:top_n],
    }
    print(json.dumps(result, indent=2))


def print_json_comparison(fg_a: FlameGraph, fg_b: FlameGraph, top_n: int, filter_pat: Optional[str]):
    pat = re.compile(filter_pat) if filter_pat else None
    label_a = Path(fg_a.path).stem
    label_b = Path(fg_b.path).stem

    all_funcs = set(fg_a.inclusive) | set(fg_b.inclusive)
    if pat:
        all_funcs = {f for f in all_funcs if pat.search(f)}

    diffs = []
    for func in all_funcs:
        pct_a = fg_a.inclusive.get(func, 0) / fg_a.total_samples * 100
        pct_b = fg_b.inclusive.get(func, 0) / fg_b.total_samples * 100
        diffs.append({
            "name": func,
            f"pct_{label_a}": round(pct_a, 2),
            f"pct_{label_b}": round(pct_b, 2),
            "diff": round(pct_b - pct_a, 2),
            f"samples_{label_a}": fg_a.inclusive.get(func, 0),
            f"samples_{label_b}": fg_b.inclusive.get(func, 0),
        })

    result = {
        label_a: {"file": fg_a.path, "total_samples": fg_a.total_samples},
        label_b: {"file": fg_b.path, "total_samples": fg_b.total_samples},
        "ratio": round(fg_b.total_samples / fg_a.total_samples, 2) if fg_a.total_samples else None,
        "by_increase": sorted(diffs, key=lambda x: -x["diff"])[:top_n],
        "by_decrease": sorted(diffs, key=lambda x: x["diff"])[:top_n],
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Parse async-profiler flamegraph HTML files")
    parser.add_argument("files", nargs="+", help="One or two .html flamegraph files")
    parser.add_argument("--filter", "-f", help="Regex filter for function names")
    parser.add_argument("--top", "-n", type=int, default=20, help="Number of top entries to show (default: 20)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--callers", "-c", help="Trace call paths to functions matching REGEX")
    parser.add_argument("--depth", "-d", type=int, default=20, help="Max stack depth to show with --callers (default: 20)")
    args = parser.parse_args()

    if args.callers:
        if len(args.files) != 1:
            parser.error("--callers requires exactly one file")
        fg = parse_flamegraph(args.files[0])
        if args.json:
            print_json_callers(fg, args.callers, args.top, args.depth)
        else:
            print_callers(fg, args.callers, args.top, args.depth)

    elif len(args.files) == 1:
        fg = parse_flamegraph(args.files[0])
        if args.json:
            print_json_single(fg, args.top, args.filter)
        else:
            print_single(fg, args.top, args.filter)

    elif len(args.files) == 2:
        fg_a = parse_flamegraph(args.files[0])
        fg_b = parse_flamegraph(args.files[1])
        if args.json:
            print_json_comparison(fg_a, fg_b, args.top, args.filter)
        else:
            print_comparison(fg_a, fg_b, args.top, args.filter)

    else:
        parser.error("Provide one file (summary) or two files (comparison)")


if __name__ == "__main__":
    main()

---
disable-model-invocation: true
description: Parse and analyze async-profiler flamegraph HTML files.
---

# Flamegraph Analysis

Parse async-profiler flamegraph HTML files to extract structured profiling data. These files encode stack traces in a compact JavaScript format (constant pool + frame calls) that browsers render as interactive flamegraphs. This skill extracts that data into tabular or JSON form for analysis.

## Parsing tool

`scripts/parse_flamegraph.py` (relative to this skill directory) is a standalone Python 3 script with no dependencies beyond the standard library.

### Single file — summarize a profile

```bash
python3 <skill-dir>/scripts/parse_flamegraph.py profile.html
python3 <skill-dir>/scripts/parse_flamegraph.py profile.html --top 30
python3 <skill-dir>/scripts/parse_flamegraph.py profile.html --filter 'is/hail'
```

### Two files — compare profiles for regressions

```bash
python3 <skill-dir>/scripts/parse_flamegraph.py before.html after.html
python3 <skill-dir>/scripts/parse_flamegraph.py before.html after.html --filter 'is/hail|__C|split_|DECODE'
```

The comparison shows:
- Total sample counts and their ratio
- Functions with the largest increase/decrease in inclusive %
- Self-time changes (where CPU time is actually spent, not just call-tree ancestry)
- Self-time excess vs uniform scaling (see below — the key regression-vs-noise discriminator)
- Functions unique to each profile

### Single file — trace call paths to a hot function

```bash
python3 <skill-dir>/scripts/parse_flamegraph.py profile.html --callers 'ArrayBuilder'
python3 <skill-dir>/scripts/parse_flamegraph.py profile.html --callers 'DECODE' --top 10
python3 <skill-dir>/scripts/parse_flamegraph.py profile.html --callers 'Region.*allocate' --depth 12
```

Shows the full call stacks leading to every frame matching the regex, grouped by unique stack and sorted by sample count. Recursive cycles (e.g., mutual recursion between `applyTable` and `lower$2`) are automatically collapsed with a repetition count (`x29`). When the matched function is itself recursive, only the topmost occurrence per stack is traced, so each sample is counted once.

### Flags

| Flag | Effect |
|------|--------|
| `--top N` / `-n N` | Show top N entries per section (default 20) |
| `--filter REGEX` / `-f REGEX` | Only show functions matching the regex (summary/comparison modes) |
| `--json` / `-j` | Output as JSON for programmatic use |
| `--callers REGEX` / `-c REGEX` | Trace call paths to functions matching the regex (single file only) |
| `--depth N` / `-d N` | Max stack depth to show with `--callers` (default 20, 0 = unlimited) |

### JSON mode

Use `--json` when you need to do further computation on the results (e.g., feeding into another script or building a report). The output is a single JSON object with `inclusive_top` and `self_time_top` arrays (single file), `by_increase`/`by_decrease`/`by_self_excess` arrays (comparison), or `stacks` array (`--callers`). In `--callers --json` mode, recursive cycles appear as `{"cycle": [...], "count": N}` objects within the frames array.

## How to interpret the output

**Inclusive time** counts every sample where a function appears anywhere in the call stack, counted once per sample: recursive functions are attributed at their topmost occurrence, so percentages are always ≤ 100%. (Naively summing frame widths would multiply a recursive walk's time by its recursion depth — a tree traversal 40 frames deep would misleadingly report 40× its true cost.)

**Self time** counts only samples where the function is at the top of the stack (the CPU was executing that function, not one of its callees). This is where the CPU is actually spending cycles. Regressions in self-time point to the actual bottleneck; regressions in inclusive-time point to the call subtree containing the bottleneck.

**Run-specific names are normalized** so the same code aligns across profiles: JVM lambda classes (`Foo$$Lambda$733/101464973` → `Foo$$Lambda$*`), reflection accessors (`GeneratedMethodAccessor120` → `GeneratedMethodAccessor*`), and Hail's generated classes/methods (`__C197...` → `__C*...`, `__m1206split_Block` → `__m*split_Block`). Compare Hail generated code by the method suffix (`split_StreamFor`, `DECODE_r_struct_of_...`, `btree_get`, ...).

**When comparing two profiles:**
- The sample ratio tells you the overall slowdown/speedup — but check whether the two captures are comparable (same capture window, same number of requests, same machine) before attributing it to code.
- The "self-time EXCESS vs uniform scaling" section is the key discriminator: it ranks functions by how much their self-time deviates from `samples_a × ratio`. A localized code regression shows a large positive excess in specific functions; if every excess is near zero, the profiles have the same shape and the ratio comes from something global — a longer capture window, more requests captured, or a slower/contended CPU — not a code change.
- Large increases in self-time % pinpoint where new work is being done.
- Functions present in only one profile reveal structural changes to the code path (renamed functions, new call wrappers, refactored APIs). One recurring false positive: `NativeMethodAccessorImpl.invoke` in one profile vs `GeneratedMethodAccessor*.invoke` in the other is JVM reflection warmup (the JVM swaps in a generated accessor after ~16 calls), not a code change.
- For JVM driver profiles, break down by thread group first: JIT compiler threads (`CompileBroker::compiler_thread_loop`) and GC workers (`GangWorker::loop`) often dominate total CPU. A regression confined to worker threads is a code change; uniform growth across JIT + GC + workers points at the environment.

## Analyzing Hail profiles specifically

Hail's JIT-compiled query code appears as `__C<N>collect_distributed_array_...` with generated method names:

| Pattern | Meaning |
|---------|---------|
| `split_StreamFor` | The main iteration loop over partitioned data |
| `split_ToArray` | Materializing a stream into an array — an extra one signals a lowering regression |
| `DECODE_*` / `INPLACE_DECODE_*` | Deserializing encoded data from Region memory |
| `btree_get` | B-tree index lookup (aggregation key lookup) |
| `compWithKey` / `ord_compare` | Key comparison during aggregation |
| `arrayref_bounds_check` | Array bounds checking in generated code |
| `position` | Stream position tracking |

Key Hail runtime functions to watch:

| Function | What it does |
|----------|--------------|
| `Region$.loadByte` / `loadBit` / `loadInt` | Low-level memory reads from Region memory |
| `Region$.storeByte` / `setBit` | Memory writes |
| `RegionMemory.allocate` / `free` | Region memory management |
| `LEB128InputBuffer.readByte` / `readInt` | Variable-length integer decoding |
| `BlockingInputBuffer.ensure` / `readByte` | Buffered I/O from encoded data |
| `LZ4.decompress` / `LZ4InputBlockBuffer.readBlock` | Block decompression |
| `GoogleStorageFS.*` | GCS I/O — check for unexpected reads/writes |
| `AnyRefArrayBuilder` / `LongArrayBuilder` | Array construction during aggregation |

## Workflow for regression analysis

1. Run the comparison: `python3 ... before.html after.html --top 30`
2. Check the sample ratio — is there an overall slowdown?
3. Check the "self-time EXCESS vs uniform scaling" section first: if no function exceeds uniform scaling, stop blaming the code — verify the captures are comparable (window length, request count, machine/CPU contention) before digging further
4. Look at self-time changes to find where new CPU work is happening
5. Filter to Hail-specific code: `--filter 'hail|__C|split_|DECODE|Region|Memory|LZ4|btree'`
6. Check for structural changes: new `split_ToArray` (materialization regression), new GCS write paths, new decode steps
7. Cross-reference with the "functions only in X" sections to find renamed/refactored code paths
8. Use `--callers` to trace call paths to any new hotspot: `--callers 'HotFunction' --depth 15`
9. Summarize findings: what changed in the generated code, what changed in the runtime, what's the likely root cause

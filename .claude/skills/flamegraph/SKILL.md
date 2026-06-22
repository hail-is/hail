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
- Functions unique to each profile

### Flags

| Flag | Effect |
|------|--------|
| `--top N` / `-n N` | Show top N entries per section (default 20) |
| `--filter REGEX` / `-f REGEX` | Only show functions matching the regex |
| `--json` / `-j` | Output as JSON for programmatic use |

### JSON mode

Use `--json` when you need to do further computation on the results (e.g., feeding into another script or building a report). The output is a single JSON object with `inclusive_top` and `self_time_top` arrays (single file), or `by_increase`/`by_decrease` arrays (comparison).

## How to interpret the output

**Inclusive time** counts every sample where a function appears anywhere in the call stack. A function called from many places will have inclusive % > 100% — this is expected and means it's a common ancestor, not that it's hot by itself.

**Self time** counts only samples where the function is at the top of the stack (the CPU was executing that function, not one of its callees). This is where the CPU is actually spending cycles. Regressions in self-time point to the actual bottleneck; regressions in inclusive-time point to the call subtree containing the bottleneck.

**When comparing two profiles:**
- The sample ratio tells you the overall slowdown/speedup.
- Large increases in self-time % pinpoint where new work is being done.
- Functions present in only one profile reveal structural changes to the code path (renamed functions, new call wrappers, refactored APIs).
- For Hail's generated code, the class number (e.g., `__C197` vs `__C194`) changes between versions — compare by method suffix (`split_StreamFor`, `DECODE_r_struct_of_...`, `btree_get`, etc.) rather than the full class name.

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
3. Look at self-time changes to find where new CPU work is happening
4. Filter to Hail-specific code: `--filter 'hail|__C|split_|DECODE|Region|Memory|LZ4|btree'`
5. Check for structural changes: new `split_ToArray` (materialization regression), new GCS write paths, new decode steps
6. Cross-reference with the "functions only in X" sections to find renamed/refactored code paths
7. Summarize findings: what changed in the generated code, what changed in the runtime, what's the likely root cause

---
title: Hail GVCF Combiner Standard Operating Procedure
author: Chris Vittal
date: 2026-09-02
updated: 2026-09-02
---

# Hail GVCF/VDS Combiner Standard Operating Procedure

## Contents

1. (Overview)[#overview]
1. (Collating Samples)[#collating-samples]
1. (Ensuring Access)[#ensuring-access]
1. (Combining)[#combining]
1. (Troubleshooting)[#troubleshooting]

## Overview

This document is intended to describe a procedure for the creation of a hail
Variant Data Set (VDS) from a collection of input GVCFs. It details best
practices as well as gives examples for using Hail Batch on GCP to perform this
work.

## Collating Samples

In order to create a joint dataset, there must first be single sample data. This
will usually be provided by a lab project manager and contain for every sample,
at minimum, the desired final sample ID (more on this in a bit), and the object
storage URI for that sample.

It is an exercise for the operator to take that data and turn it into two
lists of the same length, one a list of sample IDs and the other the list of
object URIs.

### The `new_combiner` API (part 1)

Before going further, it is useful to begin understanding the inputs to
`new_combiner`, the the public constructor for the `VariantDatasetCombiner`
class (the combiner driver).

```python3
def new_combiner(
    *,
    gvcf_paths: Optional[List[str]] = None,
    gvcf_sample_names: Optional[List[str]] = None,
    vds_paths: Optional[List[str]] = None,
    vds_sample_counts: Optional[List[int]] = None,
    ...,
) -> VariantDatasetCombiner
```

Above are the four parameters that describe the inputs to the combiner. Of
these, only one of `gvcf_paths` or `vds_paths` is actually required. Operators
of the combiner however SHOULD when providing `gvcf_paths`, provide
`gvcf_sample_names` and when providing `vds_paths`, provide `vds_sample_counts`.
This will save work on the backend as the combiner pipelines compute the sample
counts and sample names if necessary.

This is particularly important for GVCFs as often two samples across projects or
workspaces can have the same ID (and therefore there will be duplicate sample
IDs in the final VDS), but should be distinguished usually with some kind of
project prefix.

## Ensuring Access

It is _annoyingly_ common to waste time due to cloud permissions errors. Because
of this, operators should check every input both to ensure they have access to
all of them.

The following python function should suffice to check a list of URIs. When using
it, remember to include the `.tbi` tabix index file for each input GVCF.
A comprehension like `[file for gvcf in gvcfs for file in (gvcf, gvcf + '.tbi')]`
will add the corresponding index for each file.

```python3
import sys

from hailtop.aiotools.fs import AsyncFS
from hailtop.utils import bounded_gather

async def check_permissions(fs: AsyncFS, files: list[str]):
    result = await bounded_gather(*(lambda: fs.statfile(file) for file in files),
                                  parallelism=50,
                                  return_exceptions=True)
    missing_permissions = [(file, exc) for file, (_, exc) in zip(files, result) if exc is not None]
    if not missing_permissions:  # no exceptions, every file can be accessed
      return

    for file, exc in missing_permissions:
        print(f"error reading {file}, exception =", exc, file=sys.stderr)
    msg = f"Missing permissions (or files don't exist) for {len(missing_permissions)} of {len(files)} files"
    raise PermissionError(msg) from exc
```

## Combining
Once all GVCFs have been collated and permissions checked it is finally time to
run the combine job.

Actually running the combiner is as simple as:
```python3
combiner = hl.vds.new_combiner(...)
combiner.run()
```

To actually create a combiner object, we walk through each remaining portion of
the `new_combiner` constructor giving good defaults and recommendations.

### Required and Recommended Parameters

```python3
def new_combiner(
    *,
    output_path: str,
    temp_path: str,
    save_path: Optional[str] = None,
    ...,
) -> VariantDatasetCombiner:
```

The only two specifically required parameters to `new_combiner` are `output_path`
and `temp_path`. The operator MUST set `output_path` to a durable object storage
URI (no lifecycle policy on a bucket). The operator MUST set `temp_path` to
a transient object storage URI (a bucket with a lifecycle policy), however it is
RECOMMENDED that the lifecycle policy be long enough to keep all created objects
live for the duration of the whole combine job—usually up to two weeks. This
ensures that intermediates are always available while also ensuring timely
cleanup to keep storage costs down.

The operator SHOULD set `save_path` to an object storage URI, this can be either
temporary or durable. It is RECOMMENDED that it be temporary. The combiner will
produce a path based on `temp_path` and the hash of its parameters however this
is liable to change without notice so it is best to set a path for easy
introspection on the plan and ease of use.

### Import VCF parameters

```python3
def new_combiner(
    *,
    reference_genome: Union[str, ReferenceGenome] = 'default',
    contig_recoding: Optional[Dict[str, str]] = None,
    call_fields: Collection[str] = ['PGT'],
    ...,
) -> VariantDatasetCombiner:
```

The operator SHOULD set `reference_genome` to the appropriate reference genome
to avoid issues where the default does not match the reference of the samples.

To understand `contig_recoding`, refer to the [`import_vcf`] documentation. The
operator MUST set this parameter if contigs that do not match hail's
representation of the reference genome have been used for the GVCFs.

Use `call_fields` if there are fields other than `GT` and `PGT` that should be
treated as a genomic call, again refer to [`import_vcf`] for more details.

[`import_vcf`]: https://hail.is/docs/0.2/methods/impex.html#hail.methods.import_vcf

### Interval Paramters

```python3
def new_combiner(
    *,
    reference_genome: Union[str, ReferenceGenome] = 'default',
    intervals: Optional[List[Interval]] = None,
    import_interval_size: Optional[int] = None,
    use_genome_default_intervals: bool = False,
    use_exome_default_intervals: bool = False,
    ...,
```

Exactly one of the four interval parameters must be set. An error will be raised
otherwise. It is usually sufficient to use one of the `use_*_default_intervals`
paramters. These give reasonably even coverage over the genome for each sample
type (exome/genome).

When setting `intervals`, eache interval provided MUST be non-overlapping and
include both endpoints. The `intervals` SHOULD cover almost all of the canonical
contigs for the samples' reference genome (excluding some amout of
telomeres/centromeres is often ok). The operator MUST NOT use "calling
intervals" to provide `intervals`, exome calling intervals are much finer
grained and will lead to far too much parallelism.

Setting `import_interval_size`, causes the library to compute even intervals of
roughly that size to then use for GVCF import.

### Controlling Imported Fields

There are four parameters that help control what fields are kept/discarded on
GVCF import.

```python3
def new_combiner(
    *,
    gvcf_external_header: Optional[str] = None,
    gvcf_info_to_keep: Optional[Collection[str]] = None,
    gvcf_reference_entry_fields_to_keep: Optional[Collection[str]] = None,
    gvcf_save_filters: bool = False,
    ...,
) -> VariantDatasetCombiner:
```

It is RECOMMENDED that the operator set `gvcf_external_header` this saves
checking every file to make sure the types are homogeneous. This does not cause
issues since the way that VCF parsing in hail works is that fields that are not
in the header are silently skipped.

The operator SHOULD set `gvcf_save_filters` to `True` in order to save any
filters present in the GVCF, it is default `False` for backwards compatibility.

It is not usually necessary to set `gvcf_info_to_keep` or
`gvcf_reference_entry_fields_to_keep`, the combiner does a good job of
determining which fields are present in reference/variant data and keeping them.
There is one notable field that operators may wish to keep in reference data,
`PL` is dropped by default for reference data and must be explicitly added to
`gvcf_reference_entry_fields_to_keep` if it is desired to be retained.

### Parallelism Control
```python3
def new_combiner(
    *,
    branch_factor: int = 100,
    target_records: int = 24_000,
    gvcf_batch_size: int = 50,
    ...,
) -> VariantDatasetCombiner:
```

These parameters work to control how many inputs get combined at once
(`branch_factor`, default 100), how large final output partitions should be
(`target_records`, default 24,000), and how many gvcf combine jobs to attempt to
run in parallel (`gvcf_batch_size`, default 50).

These are the only parameters that do not get taken into account when saving the
combiner plan to `save_path`. This is so that they can be changed as tuning
these values is often essential to making sure the combiner finishes—more on
that later.

Operators SHOULD leave `branch_factor` and `gvcf_batch_size` as defaults and
adjust them as part of troubleshooting. It is RECOMMENDED that operators work
with downstream analysts to understand parallelism needs and set target records
accordingly, values up to a few million are not inappropriate for lower numbers
of samples (less than 10k) in a VDS.

## Troubleshooting
The combiner is often a long running pipeline. It is resilient but may need to
be tweaked or restarted.

### Resources
Combiner pipelines are very large often using lots of memory for compilation.
The jobs can fail with out of memory errors. Many strange errors of unknown
cause are resource exhaustion errors in disguise. Increasing `driver_memory` or
`worker_memory` in `hl.init` may be necessary for the jobs to proceed.

Alternatively, increasing `target_records` can decrease memory used during
compilation and execution by decreasing the number of intervals that each
intermediate VDS is read with and therefore also the size of the array of
partition results.

### Retry
The Hail Batch service has great preemption and retry logic, however it still
can't catch everything. If there is a service error or a spurious/rare cloud
error it is usually enough to restart the combine job. To do so, run the same
combine script, it will load the combiner plan and continue.

### Compilation errors
The combiner generally only sees one form of compilation error,
`ClassTooLargeException`. This usually occurs due to too many inputs and so is
most often resolved by shrinking the number of inputs at the cost of
parallelism. If this error is encountered in GCVF combining, try decreasing
`gvcf_batch_size` by 25% until the pipeline runs as expected. If there are
issues combining VDS files, try increasing `target_records` to start, especially
if there are hundreds of thousands of partitions in the VDS combine jobs. Try
doubling it to see if that makes a difference.

As a last resort to keep the combiner working, `branch_factor` can be decreased.
This parameter determines how many times the whole dataset needs to be read from
start to finish so should be kept as large as feasible. Like batch size, try
decreasing `branch_factor` by 25% until the pipeline runs again.

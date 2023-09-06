# Change Log And Version Policy

## Python Version Compatibility Policy

Hail complies with [NumPy's compatibility policy](https://numpy.org/neps/nep-0029-deprecation_policy.html#implementation) on Python
versions. In particular, Hail officially supports:

- All minor versions of Python released 42 months prior to the project, and at minimum the two
  latest minor versions.

- All minor versions of numpy released in the 24 months prior to the project, and at minimum the
  last three minor versions.

## Frequently Asked Questions

### With a version like 0.x, is Hail ready for use in publications?

Yes. The [semantic versioning standard](https://semver.org) uses 0.x (development) versions to
refer to software that is either "buggy" or "partial". While we don't view
Hail as particularly buggy (especially compared to one-off untested
scripts pervasive in bioinformatics!), Hail 0.2 is a partial realization
of a larger vision.

### What is the difference between the Hail Python library version and the native file format version?

The Hail Python library version, the version you see on [PyPI](https://pypi.org/project/hail/), in
`pip`, or in `hl.version()` changes every time we release the Python library. The Hail native file
format version only changes when we change the format of Hail Table and MatrixTable files. If a
version of the Python library introduces a new native file format version, we note that in the
change log. All subsequent versions of the Python library can read the new file format version.

The native file format changes much slower than the Python library version. It is not currently
possible to view the file format version of a Hail Table or MatrixTable.

### What stability is guaranteed?

The Hail file formats and Python API are backwards compatible. This means that a script developed to
run on Hail 0.2.5 should continue to work in every subsequent release within the 0.2 major version.
This also means any file written by python library versions 0.2.1 through 0.2.5 can be read by
0.2.5.

Forward compatibility of file formats and the Python API is not guaranteed. In particular, a new
file format version is only readable by library versions released after the file format. For
example, Python library version 0.2.119 introduces a new file format version: 1.7.0. All library
versions before 0.2.119, for example 0.2.118, *cannot* read file format version 1.7.0. All library
versions after and including 0.2.119 *can* read file format version 1.7.0.

Each version of the Hail Python library can only write files using the latest file format version it
supports.

**The hl.experimental package and other methods marked experimental in the docs are exempt from this
policy. Their functionality or even existence may change without notice. Please contact us if you
critically depend on experimental functionality.**

## Version 0.2.121

Released 2023-08-31

### New Features

- (hail#13385) The VDS combiner now supports arbitrary custom call fields via the `call_fields`
  parameter.
- (hail#13224) `hailctl config get`, `set`, and `unset` now support shell auto-completion. Run
  `hailctl --install-completion zsh` to install the auto-completion for `zsh`. You must already have
  completion enabled for `zsh`.
- (hail#13279) Add `hailctl batch init` which helps new users interactively set up `hailctl` for
  Query-on-Batch and Batch use.

### Bug Fixes
- (hail#13573) Fix (hail#12936) in which VEP frequently failed (due to Docker not starting up) on
  clusters with a non-trivial number of workers.
- (hail#13485) Fix (hail#13479) in which `hl.vds.local_to_global` could produce invalid values when
  the LA field is too short. There were and are no issues when the LA field has the correct length.
- (hail#13340) Fix `copy_log` to correctly copy relative file paths.
- (hail#13364) `hl.import_gvcf_interval` now treats `PGT` as a call field.
- (hail#13333) Fix interval filtering regression: `filter_rows` or `filter` mentioning the same
  field twice or using two fields incorrectly read the entire dataset. In 0.2.121, these filters
  will correctly read only the relevant subset of the data.
- (hail#13368) In Azure, Hail now uses fewer "list blobs" operations. This should reduce cost on
  pipelines that import many files, export many of files, or use file glob expressions.
- (hail#13414) Resolves (hail#13407) in which uses of `union_rows` could reduce parallelism to one
  partition resulting in severely degraded performance.
- (hail#13405) `MatrixTable.aggregate_cols` no longer forces a distributed computation. This should
  be what you want in the majority of cases. In case you know the aggregation is very slow and
  should be parallelized, use `mt.cols().aggregate` instead.
- (hail#13460) In Query-on-Spark, restore `hl.read_table` optimization that avoids reading
  unnecessary data in pipelines that do not reference row fields.
- (hail#13447) Fix (hail#13446). In all three submit commands (`batch`, `dataproc`, and
  `hdinsight`), Hail now allows and encourages the use of -- to separate arguments meant for the
  user script from those meant for hailctl. In hailctl batch submit, option-like arguments, for
  example "--foo", are now supported before "--" if and only if they do not conflict with a hailctl
  option.
- (hail#13422) `hailtop.hail_frozenlist.frozenlist` now has an eval-able `repr`.
- (hail#13523) `hl.Struct` is now pickle-able.
- (hail#13505) Fix bug introduced in 0.2.117 by commit `c9de81108` which prevented the passing of
  keyword arguments to Python jobs. This manifested as "ValueError: too many values to unpack".
- (hail#13536) Fixed (hail#13535) which prevented the use of Python jobs when the client (e.g. your
  laptop) Python version is 3.11 or later.
- (hail#13434) In QoB, Hail's file systems now correctly list all files in a directory, not just the
  first 1000. This could manifest in an `import_table` or `import_vcf` which used a glob
  expression. In such a case, only the first 1000 files would have been included in the resulting
  Table or MatrixTable.
- (hail#13550) `hl.utils.range_table(n)` now supports all valid 32-bit signed integer values of `n`.
- (hail#13500) In Query-on-Batch, the client-side Python code will not try to list every job when a
  QoB batch fails. This could take hours for long-running pipelines or pipelines with many
  partitions.


### Deprecations

- (hail#13275) Hail no longer officially supports Python 3.8.

## Version 0.2.120

Released 2023-07-20

### New Features
- (hail#13206) The VDS Combiner now works in Query-on-Batch.

### Bug Fixes
- (hail#13313) Fix bug introduced in 0.2.119 which causes a serialization error when using
  Query-on-Spark to read a VCF which is sorted by locus, with split multi-allelics, in which the
  records sharing a single locus do not appear in the dictionary ordering of their alternate
  alleles.
- (hail#13264) Fix bug which ignored the `partition_hint` of a Table group-by-and-aggregate.
- (hail#13239) Fix bug which ignored the `HAIL_BATCH_REGIONS` argument when determining in which
  regions to schedule jobs when using Query-on-Batch.
- (hail#13253) Improve `hadoop_ls` and `hfs.ls` to quickly list globbed files in a directory. The
  speed improvement is proportional to the number of files in the directory.
- (hail#13226) Fix the comparison of an `hl.Struct` to an `hl.struct` or field of type
  `tstruct`. Resolves (hail#13045) and (Hail#13046).
- (hail#12995) Fixed bug causing poor performance and memory leaks for `MatrixTable.annotate_rows`
  aggregations.

## Version 0.2.119

Released 2023-06-28

### New Features
- (hail#12081) Hail now uses [Zstandard](https://facebook.github.io/zstd/) as
  the default compression algorithm for table and matrix table storage. Reducing
  file size around 20% in most cases.
- (hail#12988) Arbitrary aggregations can now be used on arrays via
  `ArrayExpression.aggregate`. This method is useful for accessing
  functionality that exists in the aggregator library but not the
  basic expression library, for instance, `call_stats`.
- (hail#13166) Add an `eigh` ndarray method, for finding eigenvalues
  of symmetric matrices ("h" is for Hermitian, the complex analogue of
  symmetric).

### Bug Fixes
- (hail#13184) The `vds.to_dense_mt` no longer densifies past the end of
  contig boundaries. A logic bug in `to_dense_mt` could lead to reference data
  toward's the end of one contig being applied to the following contig up until
  the first reference block of the contig.
- (hail#13173) Fix globbing in scala blob storage filesystem implementations.

### File Format

- The native file format version is now 1.7.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

## Version 0.2.118

Released 2023-05-30

### New Features

- (hail#13140) Enable `hail-az` and Azure Blob Storage `https` URLs to contain SAS tokens to enable bearer-auth style file access to Azure storage.
- (hail#13129) Allow subnet to be passed through to gcloud in hailctl.

### Bug Fixes

- (hail#13126) Query-on-Batch pipelines with one partition are now retried when they encounter transient errors.
- (hail#13113) `hail.ggplot.geom_point` now displays a legend group for a column even when it has only one value in it.
- (hail#13075) (hail#13074) Add a new transient error plaguing pipelines in Query-on-Batch in Google: `java.net.SocketTimeoutException: connect timed out`.
- (hail#12569) The documentation for `hail.ggplot.facets` is now correctly included in the API reference.

---

## Version 0.2.117

Released 2023-05-19

### New Features

- (hail#12875) Parallel export modes now write a manifest file. These manifest files are text files with one filename per line, containing name of each shard written successfully to the directory. These filenames are relative to the export directory.
- (hail#13007) In Query-on-Batch and `hailtop.batch`, memory and storage request strings may now be optionally terminated with a `B` for bytes.

### Bug Fixes

- (hail#13065) In Azure Query-on-Batch, fix a resource leak that prevented running pipelines with >500 partitions and created flakiness with >250 partitions.
- (hail#13067) In Query-on-Batch, driver and worker logs no longer buffer so messages should arrive in the UI after a fixed delay rather than proportional to the frequency of log messages.
- (hail#13028) Fix crash in `hl.vds.filter_intervals` when using a table to filter a VDS that stores the max ref block length.
- (hail#13060) Prevent 500 Internal Server Error in Jupyter Notebooks of Dataproc clusters started by `hailctl dataproc`.
- (hail#13051) In Query-on-Batch and `hailtop.batch`, Azure Blob Storage `https` URLs are now supported.
- (hail#13042) In Query-on-Batch, `naive_coalesce` no longer performs a full write/read of the dataset. It now operates identically to the Query-on-Spark implementation.
- (hail#13031) In `hl.ld_prune`, an informative error message is raised when a dataset does not contain diploid calls instead of an assertion error.
- (hail#13032) In Query-on-Batch, in Azure, Hail now users a newer version of the Azure blob storage libraries to reduce the frequency of "Stream is already closed" errors.
- (hail#13011) In Query-on-Batch, the driver will use ~1/2 as much memory to read results as it did in 0.2.115.
- (hail#13013) In Query-on-Batch, transient errors while streaming from Google Storage are now automatically retried.

---

## Version 0.2.116

Released 2023-05-08

### New Features

- (hail#12917) ABS blob URIs in the format of `https://<ACCOUNT_NAME>.blob.core.windows.net/<CONTAINER_NAME>/<PATH>` are now supported.
- (hail#12731) Introduced `hailtop.fs` that makes public a filesystem module that works for local fs, gs, s3 and abs. This is now used as the `Backend.fs` for hail query but can be used standalone for Hail Batch users by `import hailtop.fs as hfs`.

### Deprecations

- (hail#12929) Hail no longer officially supports Python 3.7.
- (hail#12917) The `hail-az` scheme for referencing blobs in ABS is now deprecated and will be removed in an upcoming release.

### Bug Fixes

- (hail#12913) Fixed bug in `hail.ggplot` where all legend entries would have the same text if one column had exactly one value for all rows and was mapped to either the `shape` or the `color` aesthetic for `geom_point`.
- (hail#12901) `hl.Struct` now has a correct and useful implementation of `pprint`.

---

## Version 0.2.115

Released 2023-04-25

### New Features

- (hail#12731) Introduced `hailtop.fs` that makes public a filesystem module that works
  for local fs, gs, s3 and abs. This can be used by `import hailtop.fs as hfs` but has also
  replaced the underlying implementation of the `hl.hadoop_*` methods. This means that the
  `hl.hadoop_*` methods now support these additional blob storage providers.
- (hail#12917) ABS blob URIs in the form of `https://<ACCOUNT_NAME>.blob.core.windows.net/<CONTAINER_NAME>/<PATH>` are now supported when running in Azure.

### Deprecations
- (hail#12917) The `hail-az` scheme for referencing ABS blobs in Azure is deprecated in favor of the `https` scheme and will be removed in a future release.

### Bug Fixes

- (hail#12919) An interactive hail session is no longer unusable after hitting CTRL-C during a batch execution in Query-on-Batch
- (hail#12913) Fixed bug in `hail.ggplot` where all legend entries would have the same text if one column had exactly one value for all rows and was mapped to either the `shape` or the `color` aesthetic for `geom_point`.

---

## Version 0.2.114

Released 2023-04-19

### New Features

- (hail#12880) Added `hl.vds.store_ref_block_max_len` to patch old VDSes to make interval filtering faster.

### Bug Fixes

- (hail#12860) Fixed memory leak in shuffles in Query-on-Batch.

---

## Version 0.2.113

Released 2023-04-07

### New Features

- (hail#12798) Query-on-Batch now supports `BlockMatrix.write(..., stage_locally=True)`.
- (hail#12793) Query-on-Batch now supports `hl.poisson_regression_rows`.
- (hail#12801) Hitting CTRL-C while interactively using Query-on-Batch cancels the underlying batch.
- (hail#12810) `hl.array` can now convert 1-d ndarrays into the equivalent list.
- (hail#12851) `hl.variant_qc` no longer requires a locus field.
- (hail#12816) In Query-on-Batch, `hl.logistic_regression('firth', ...)` is now supported.
- (hail#12854) In Query-on-Batch, simple pipelines with large numbers of partitions should be substantially faster.


### Bug Fixes

- (hail#12783) Fixed bug where logs were not properly transmitted to Python.
- (hail#12812) Fixed bug where `Table/MT._calculate_new_partitions` returned unbalanced intervals with whole-stage code generation runtime.
- (hail#12839) Fixed `hailctl dataproc` jupyter notebooks to be compatible with Spark 3.3, which have been broken since 0.2.110.
- (hail#12855) In Query-on-Batch, allow writing to requester pays buckets, which was broken before this release.

---

## Version 0.2.112

Released 2023-03-15

### Bug Fixes

- (hail#12784) Removed an internal caching mechanism in Query on Batch that caused stalls in pipelines with large intermediates

---

## Version 0.2.111

Released 2023-03-13

### New Features

- (hail#12581) In Query on Batch, users can specify which regions to have jobs run in.


### Bug Fixes

- (hail#12772) Fix `hailctl hdinsight submit` to pass args to the files

---

## Version 0.2.110

Released 2023-03-08

### New Features

- (hail#12643) In Query on Batch, `hl.skat(..., logistic=True)` is now supported.
- (hail#12643) In Query on Batch, `hl.liftover` is now supported.
- (hail#12629) In Query on Batch, `hl.ibd` is now supported.
- (hail#12722) Add `hl.simulate_random_mating` to generate a population from founders under the assumption of random mating.
- (hail#12701) Query on Spark now officially supports Spark 3.3.0 and Dataproc 2.1.x

### Performance Improvements

- (hail#12679) In Query on Batch, `hl.balding_nichols_model` is slightly faster. Also added `hl.utils.genomic_range_table` to quickly create a table keyed by locus.

### Bug Fixes

- (hail#12711) In Query on Batch, fix null pointer exception (manifesting as `scala.MatchError: null`) when reading data from requester pays buckets.
- (hail#12739) Fix `hl.plot.cdf`, `hl.plot.pdf`, and `hl.plot.joint_plot` which were broken by changes in Hail and changes in bokeh.
- (hail#12735) Fix (hail#11738) by allowing user to override default types in `to_pandas`.
- (hail#12760) Mitigate some JVM bytecode generation errors, particularly those related to too many method parameters.
- (hail#12766) Fix (hail#12759) by loosening `parsimonious` dependency pin.
- (hail#12732) In Query on Batch, fix bug that sometimes prevented terminating a pipeline using Control-C.
- (hail#12771) Use a version of `jgscm` whose version complies with PEP 440.

---

## Version 0.2.109

Released 2023-02-08

### New Features

- (hail#12605) Add `hl.pgenchisq` the cumulative distribution function of the generalized chi-squared distribution.
- (hail#12637) Query-on-Batch now supports `hl.skat(..., logistic=False)`.
- (hail#12645) Added `hl.vds.truncate_reference_blocks` to transform a VDS to checkpoint reference blocks in order to drastically improve interval filtering performance. Also added `hl.vds.merge_reference_blocks` to merge adjacent reference blocks according to user criteria to better compress reference data.

### Bug Fixes

- (hail#12650) Hail will now throw an exception on `hl.export_bgen` when there is no GP field, instead of exporting null records.
- (hail#12635) Fix bug where `hl.skat` did not work on Apple M1 machines.
- (hail#12571) When using Query-on-Batch, hl.hadoop* methods now properly support creation and modification time.
- (hail#12566) Improve error message when combining incompatibly indexed fields in certain operations including array indexing.

---

## Version 0.2.108

Released 2023-1-12

### New Features

- (hail#12576) `hl.import_bgen` and `hl.export_bgen` now support compression with Zstd.

### Bug fixes

- (hail#12585) `hail.ggplot`s that have more than one legend group or facet are now interactive. If such a plot has enough legend entries that the legend would be taller than the plot, the legend will now be scrollable. Legend entries for such plots can be clicked to show/hide traces on the plot, but this does not work and is a known issue that will only be addressed if `hail.ggplot` is migrated off of plotly.
- (hail#12584) Fixed bug which arose as an assertion error about type mismatches. This was usually triggered when working with tuples.
- (hail#12583) Fixed bug which showed an empty table for `ht.col_key.show()`.
- (hail#12582) Fixed bug where matrix tables with duplicate col keys do not show properly. Also fixed bug where tables and matrix tables with HTML unsafe column headers are rendered wrong in Jupyter.
- (hail#12574) Fixed a memory leak when processing tables. Could trigger unnecessarily high memory use and out of memory errors when there are many rows per partition or large key fields.
- (hail#12565) Fixed a bug that prevented exploding on a field of a Table whose value is a random value.

---

## Version 0.2.107

Released 2022-12-14

### Bug fixes

- (hail#12543) Fixed `hl.vds.local_to_global` error when LA array contains non-ascending allele indices.

---

## Version 0.2.106

Released 2022-12-13

### New Features

- (hail#12522) Added `hailctl` config setting `'batch/backend'` to specify the default backend to use in batch scripts when not specified in code.
- (hail#12497) Added support for `scales`, `nrow`, and `ncol` arguments, as well as grouped legends, to `hail.ggplot.facet_wrap`.
- (hail#12471) Added `hailctl batch submit` command to run local scripts inside batch jobs.
- (hail#12525) Add support for passing arguments to `hailctl batch submit`.
- (hail#12465) Batch jobs' status now contains the region the job ran in. The job itself can access which region it is in through the `HAIL_REGION` environment variable.
- (hail#12464) When using Query-on-Batch, all jobs for a single hail session are inserted into the same batch instead of one batch per action.
- (hail#12457) `pca` and `hwe_normalized_pca` are now supported in Query-on-Batch.
- (hail#12376) Added `hail.query_table` function for reading tables with indices from Python.
- (hail#12139) Random number generation has been updated, but shouldn't affect most users. If you need to manually set seeds, see https://hail.is/docs/0.2/functions/random.html for details.
- (hail#11884) Added `Job.always_copy_output` when using the `ServiceBackend`. The default behavior is `False`, which is a breaking change from the previous behavior to always copy output files regardless of the job's completion state.
- (hail#12139) Brand new random number generation, shouldn't affect most users. If you need to manually set seeds, see https://hail.is/docs/0.2/functions/random.html for details.

### Bug Fixes
- (hail#12487) Fixed a bug causing rare but deterministic job failures deserializing data in Query-on-Batch.
- (hail#12535) QoB will now error if the user reads from and writes to the same path. QoB also now respects the user's configuration of `disable_progress_bar`. When `disable_progress_bar` is unspecified, QoB only disables the progress bar for non-interactive sessions.
- (hail#12517) Fix a performance regression that appears when using `hl.split_multi_hts` among other methods.

---

## Version 0.2.105

Released 2022-10-31 🎃

### New Features

- (hail#12293)  Added support for `hail.MatrixTable`s to `hail.ggplot`.

### Bug Fixes

- (hail#12384) Fixed a critical bug that disabled tree aggregation and scan executions in 0.2.104, leading to out-of-memory errors.
- (hail#12265) Fix long-standing bug wherein `hl.agg.collect_as_set` and `hl.agg.counter` error when applied to types which, in Python, are unhashable. For example, `hl.agg.counter(t.list_of_genes)` will not error when `t.list_of_genes` is a list. Instead, the counter dictionary will use `FrozenList` keys from the `frozenlist` package.

---

## Version 0.2.104

Release 2022-10-19

### New Features

- (hail#12346): Introduced new progress bars which include total time elapsed and look cool.

---

## Version 0.2.103

Release 2022-10-18

### Bug Fixes

- (hail#12305): Fixed a rare crash reading tables/matrixtables with _intervals

---

## Version 0.2.102

Released 2022-10-06

### New Features

- (hail#12218) Missing values are now supported in primitive columns in `Table.to_pandas`.
- (hail#12254) Cross-product-style legends for data groups have been replaced with factored ones (consistent with `ggplot2`'s implementation) for `hail.ggplot.geom_point`, and support has been added for custom legend group labels.
- (hail#12268) `VariantDataset` now implements `union_rows` for combining datasets with the same samples but disjoint variants.

### Bug Fixes

- (hail#12278) Fixed bug made more likely by 0.2.101 in which Hail errors when interacting with a NumPy integer or floating point type.
- (hail#12277) Fixed bug in reading tables/matrixtables with partition intervals that led to error or segfault.

---

## Version 0.2.101

Released 2022-10-04

### New Features

- (hail#12218) Support missing values in primitive columns in `Table.to_pandas`.
- (hail#12195) Add a `impute_sex_chr_ploidy_from_interval_coverage` to impute sex ploidy directly from a coverage MT.
- (hail#12222) Query-on-Batch pipelines now add worker jobs to the same batch as the driver
job instead of producing a new batch per stage.
- (hail#12244) Added support for custom labels for per-group legends to `hail.ggplot.geom_point` via the
`legend_format` keyword argument

### Deprecations

- (hail#12230) The python-dill Batch images in `gcr.io/hail-vdc` are no longer supported.
Use `hailgenetics/python-dill` instead.

### Bug fixes

- (hail#12215) Fix search bar in the Hail Batch documentation.

---

## Version 0.2.100

Released 2022-09-23

### New Features

- (hail#12207) Add support for the `shape` aesthetic to `hail.ggplot.geom_point`.

### Deprecations

- (hail#12213) The `batch_size` parameter of `vds.new_combiner` is deprecated in favor of `gvcf_batch_size`.

### Bug fixes

- (hail#12216) Fix bug that caused `make install-on-cluster` to fail with a message about `sys_platform`.
- (hail#12164) Fix bug that caused Query on Batch pipelines to fail on datasets with indexes greater than 2GiB.

---

## Version 0.2.99

Released 2022-09-13

### New Features

- (hail#12091) Teach `Table` to `write_many`, which writes one table per provided field.
- (hail#12067) Add `rand_int32` and `rand_int64` for generating random 32-bit and 64-bit integers, respectively.

### Performance Improvements

- (hail#12159) Improve performance of MatrixTable reads when using `_intervals` argument

### Bug fixes

- (hail#12179) Fix incorrect composition of interval filters with unordered interval lists that could lead to over- or under-filtering.
- (hail#12162) Fixed crash in `collect_cols_by_key` with preceding random functions.

---

## Version 0.2.98

Released 2022-08-22

### New Features

- (hail#12062) `hl.balding_nichols_model` now supports an optional boolean parameter, `phased`, to control the phasedness of the generated genotypes.

### Performance improvements

- (hail#12099) Make repeated VCF/PLINK queries much faster by caching compiler data structures.
- (hail#12038) Speed up `hl.import_matrix_table` by caching header line computation.

### Bug fixes

- (hail#12115) When using `use_new_shuffle=True`, fix a bug when there are more than 2^31 rows
- (hail#12074) Fix bug where `hl.init` could silently overwrite the global random seed.
- (hail#12079) Fix bug in handling of missing (aka NA) fields in grouped aggregation and distinct by key.
- (hail#12056) Fix `hl.export_vcf` to actually create tabix files when requested.
- (hail#12020) Fix bug in `hl.experimental.densify` which manifested as an `AssertionError` about dtypes.

---

## Version 0.2.97

Released 2022-06-30

### New Features

- (hail#11756) `hb.BatchPoolExecutor` and Python jobs both now also support async functions.

### Bug fixes

- (hail#11962) Fix error (logged as (hail#11891)) in VCF combiner when exactly 10 or 100 files are combined.
- (hail#11969) Fix `import_table` and `import_lines` to use multiple partitions when `force_bgz` is used.
- (hail#11964) Fix erroneous "Bucket is a requester pays bucket but no user project provided." errors in Google Dataproc by updating to the latest Dataproc image version.

---

## Version 0.2.96

Released 2022-06-21

### New Features

- (hail#11833) `hl.rand_unif` now has default arguments of 0.0 and 1.0

### Bug fixes

- (hail#11905) Fix erroneous FileNotFoundError in glob patterns
- (hail#11921) and (hail#11910) Fix file clobbering during text export with speculative execution.
- (hail#11920) Fix array out of bounds error when tree aggregating a multiple of 50 partitions.
- (hail#11937) Fixed correctness bug in scan order for `Table.annotate` and `MatrixTable.annotate_rows` in certain circumstances.
- (hail#11887) Escape VCF description strings when exporting.
- (hail#11886) Fix an error in an example in the docs for `hl.split_multi`.

---

## Version 0.2.95

Released 2022-05-13

### New features

- (hail#11809) Export `dtypes_from_pandas` in `expr.types`
- (hail#11807) Teach smoothed_pdf to add a plot to an existing figure.
- (hail#11746) The ServiceBackend, in interactive mode, will print a link to the currently executing driver batch.
- (hail#11759) `hl.logistic_regression_rows`, `hl.poisson_regression_rows`, and `hl.skat` all now support configuration of the maximum number of iterations and the tolerance.
- (hail#11835) Add `hl.ggplot.geom_density` which renders a plot of an approximation of the probability density function of its argument.

### Bug fixes

- (hail#11815) Fix incorrectly missing entries in to_dense_mt at the position of ref block END.
- (hail#11828) Fix `hl.init` to not ignore its `sc` argument. This bug was introduced in 0.2.94.
- (hail#11830) Fix an error and relax a timeout which caused `hailtop.aiotools.copy` to hang.
- (hail#11778) Fix a (different) error which could cause hangs in `hailtop.aiotools.copy`.

---

## Version 0.2.94

Released 2022-04-26

### Deprecation

- (hail#11765) Deprecated and removed linear mixed model functionality.

### Beta features

- (hail#11782) `hl.import_table` is up to twice as fast for small tables.

### New features

- (hail#11428) `hailtop.batch.build_python_image` now accepts a `show_docker_output` argument to toggle printing docker's output to the terminal while building container images
- (hail#11725) `hl.ggplot` now supports `facet_wrap`
- (hail#11776) `hailtop.aiotools.copy` will always show a progress bar when `--verbose` is passed.

### `hailctl dataproc`
- (hail#11710) support pass-through arguments to `connect`

### Bug fixes

 - (hail#11792) Resolved issue where corrupted tables could be created with whole-stage code generation enabled.

---

## Version 0.2.93

Release 2022-03-27

### Beta features

- Several issues with the beta version of Hail Query on Hail Batch are addressed in this release.

---

## Version 0.2.92

Release 2022-03-25

### New features

- (hail#11613) Add `hl.ggplot` support for `scale_fill_hue`, `scale_color_hue`, and `scale_fill_manual`,
  `scale_color_manual`. This allows for an infinite number of discrete colors.
- (hail#11608) Add all remaining and all versions of extant public gnomAD datasets to the Hail
  Annotation Database and Datasets API. Current as of March 23rd 2022.
- (hail#11662) Add the `weight` aesthetic `geom_bar`.

### Beta features

- This version of Hail includes all the necessary client-side infrastructure to execute Hail Query
  pipelines on a Hail Batch cluster. This effectively enables a "serverless" version of Hail Query
  which is independent of Apache Spark. Broad affiliated users should contact the Hail team for help
  using Hail Query on Hail Batch. Unaffiliated users should also contact the Hail team to discuss
  the feasibility of running your own Hail Batch cluster. The Hail team is accessible at both
  https://hail.zulipchat.com and https://discuss.hail.is .

---

## Version 0.2.91

Release 2022-03-18

### Bug fixes

- (hail#11614) Update `hail.utils.tutorial.get_movie_lens` to use `https` instead of `http`. Movie
  Lens has stopped serving data over insecure HTTP.
- (hail#11563) Fix issue [hail-is/hail#11562](https://github.com/hail-is/hail/issues/11562).
- (hail#11611) Fix a bug that prevents the display of `hl.ggplot.geom_hline` and
  `hl.ggplot.geom_vline`.

---

## Version 0.2.90

Release 2022-03-11

### Critical BlockMatrix from_numpy correctness bug

- (hail#11555) `BlockMatrix.from_numpy` did not work correctly. Version 1.0 of org.scalanlp.breeze, a dependency of Apache Spark
that hail also depends on, has a correctness bug that results in BlockMatrices that repeat the top left block of the block
matrix for every block. This affected anyone running Spark 3.0.x or 3.1.x.

### Bug fixes

- (hail#11556) Fixed assertion error ocassionally being thrown by valid joins where the join key was a prefix of the left key.

### Versioning

- (hail#11551) Support Python 3.10.

---

## Version 0.2.89

Release 2022-03-04

- (hail#11452) Fix `impute_sex_chromosome_ploidy` docs.

---

## Version 0.2.88

Release 2022-03-01

This release addresses the deploy issues in the 0.2.87 release of Hail.

---

## Version 0.2.87

Release 2022-02-28

An error in the deploy process required us to yank this release from PyPI. Please do not use this
release.

### Bug fixes

- (hail#11401) Fixed bug where `from_pandas` didn't support missing strings.

---

## Version 0.2.86

Release 2022-02-25

### Bug fixes

- (hail#11374) Fixed bug where certain pipelines that read in PLINK files would give assertion error.
- (hail#11401) Fixed bug where `from_pandas` didn't support missing ints.

### Performance improvements

- (hail#11306) Newly written tables that have no duplicate keys will be faster to join against.

---

## Version 0.2.85

Release 2022-02-14

### Bug fixes

- (hail#11355) Fixed assertion errors being hit relating to RVDPartitioner.
- (hail#11344) Fix error where hail ggplot would mislabel points after more than 10 distinct colors were used.

### New features

- (hail#11332) Added `geom_ribbon` and `geom_area` to hail ggplot.

---

## Version 0.2.84

Release 2022-02-10

### Bug fixes

- (hail#11328) Fix bug where occasionally files written to disk would be unreadable.
- (hail#11331) Fix bug that potentially caused files written to disk to be unreadable.
- (hail#11312) Fix aggregator memory leak.
- (hail#11340) Fix bug where repeatedly annotating same field name could cause failure to compile.
- (hail#11342) Fix to possible issues about having too many open file handles.

### New features

- (hail#11300) `geom_histogram` infers min and max values automatically.
- (hail#11317) Add support for `alpha` aesthetic and `identity` position to `geom_histogram`.

---

## Version 0.2.83

Release 2022-02-01

### Bug fixes

- (hail#11268) Fixed `log` argument in `hail.plot.histogram`.
- (hail#11276) Fixed `log` argument in `hail.plot.pdf`.
- (hail#11256) Fixed memory leak in LD Prune.

### New features

- (hail#11274) Added `geom_col` to `hail.ggplot`.

### hailctl dataproc

- (hail#11280) Updated dataproc image version to one not affected by log4j vulnerabilities.

---

## Version 0.2.82

Release 2022-01-24

### Bug fixes

- (hail#11209) Significantly improved usefulness and speed of `Table.to_pandas`, resolved several bugs with output.

### New features

- (hail#11247) Introduces a new experimental plotting interface `hail.ggplot`, based on R's ggplot library.
- (hail#11173) Many math functions like `hail.sqrt` now automatically broadcast over ndarrays.

### Performance Improvements

- (hail#11216) Significantly improve performance of `parse_locus_interval`

### Python and Java Support

- (hail#11219) We no longer officially support Python 3.6, though it may continue to work in the short term.
- (hail#11220) We support building hail with Java 11.

### File Format

- The native file format version is now 1.6.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

---

## Version 0.2.81

Release 2021-12-20

### hailctl dataproc

- (hail#11182) Updated Dataproc image version to mitigate yet more Log4j vulnerabilities.

---

## Version 0.2.80

Release 2021-12-15

### New features

- (hail#11077) `hl.experimental.write_matrix_tables` now returns the paths of the written matrix tables.

### hailctl dataproc

- (hail#11157) Updated Dataproc image version to mitigate the Log4j vulnerability.
- (hail#10900) Added `--region` parameter to `hailctl dataproc submit`.
- (hail#11090) Teach `hailctl dataproc describe` how to read URLs with the protocols `s3` (Amazon S3), `hail-az` (Azure Blob Storage), and `file` (local file system) in addition to `gs` (Google Cloud Storage).

---

## Version 0.2.79

Release 2021-11-17

### Bug fixes

- (hail#11023) Fixed bug in call decoding that was introduced in version 0.2.78.

### New features

- (hail#10993) New function `p_value_excess_het`.

---

## Version 0.2.78

Release 2021-10-19

### Bug fixes
- (hail#10766) Don't throw out of memory error when broadcasting more than 2^(31) - 1 bytes.
- (hail#10910) Filters on key field won't be slowed down by uses of `MatrixTable.localize_entries` or `Table.rename`.
- (hail#10959) Don't throw an error in certain situations where some key fields are optimized away.

### New features
- (hail#10855) Arbitrary aggregations can be implemented using `hl.agg.fold`.


### Performance Improvements
- (hail#10971) Substantially improve the speed of `Table.collect` when collecting large amounts of data.

---

## Version 0.2.77

Release 2021-09-21

### Bug fixes

- (hail#10888) Fix crash when calling `hl.liftover`.
- (hail#10883) Fix crash / long compilation times writing matrix tables with many partitions.

---

## Version 0.2.76

Released 2021-09-15

### Bug fixes

- (hail#10872) Fix long compile times or method size errors when writing tables with many partitions
- (hail#10878) Fix crash importing or sorting tables with empty data partitions

---

## Version 0.2.75

Released 2021-09-10

### Bug fixes

- (hail#10733) Fix a bug in tabix parsing when the size of the list of all sequences is large.
- (hail#10765) Fix rare bug where valid pipelines would fail to compile if intervals were created conditionally.
- (hail#10746) Various compiler improvements, decrease likelihood of `ClassTooLarge` errors.
- (hail#10829) Fix a bug where `hl.missing` and `CaseBuilder.or_error` failed if their type was a struct containing a field starting with a number.

### New features

- (hail#10768) Support multiplying `StringExpression`s to repeat them, as with normal python strings.

### Performance improvements

- (hail#10625) Reduced need to copy strings around, pipelines with many string operations should get faster.
- (hail#10775) Improved performance of `to_matrix_table_row_major` on both `BlockMatrix` and `Table`.

---

## Version 0.2.74

Released 2021-07-26

### Bug fixes

- (hail#10697) Fixed bug in `read_table` when the table has missing keys and `_n_partitions` is specified.
- (hail#10695) Fixed bug in hl.experimental.loop causing incorrect results when loop state contained pointers.

---


## Version 0.2.73

Released 2021-07-22

### Bug fixes

- (hail#10684) Fixed a rare bug reading arrays from disk where short arrays would have their first elements corrupted and long arrays would cause segfaults.
- (hail#10523) Fixed bug where liftover would fail with "Could not initialize class" errors.

---

## Version 0.2.72

Released 2021-07-19

### New Features

- (hail#10655) Revamped many hail error messages to give useful python stack traces.
- (hail#10663) Added `DictExpression.items()` to mirror python's `dict.items()`.
- (hail#10657) `hl.map` now supports mapping over multiple lists like Python's built-in `map`.

### Bug fixes

- (hail#10662) Fixed partitioning logic in `hl.import_plink`.
- (hail#10669) `NDArrayNumericExpression.sum()` now works correctly on ndarrays of booleans.

---

## Version 0.2.71

Released 2021-07-08

### New Features

- (hail#10632) Added support for weighted linear regression to `hl.linear_regression_rows`.
- (hail#10635) Added `hl.nd.maximum` and `hl.nd.minimum`.
- (hail#10602) Added `hl.starmap`.

### Bug fixes

- (hail#10038) Fixed crashes when writing/reading matrix tables with 0 partitions.
- (hail#10624) Fixed out of bounds bug with `_quantile_from_cdf`.


### hailctl dataproc

- (hail#10633) Added `--scopes` parameter to `hailctl dataproc start`.

---

## Version 0.2.70

Released 2021-06-21

---

## Version 0.2.69

Released 2021-06-14

### New Features

- (hail#10592) Added `hl.get_hgdp` function.
- (hail#10555) Added `hl.hadoop_scheme_supported` function.
- (hail#10551) Indexing ndarrays now supports ellipses.

### Bug fixes

- (hail#10553) Dividing two integers now returns a `float64`, not a `float32`.
- (hail#10595) Don't include nans in `lambda_gc_agg`.

### hailctl dataproc

- (hail#10574) Hail logs will now be stored in `/home/hail` by default.

---

## Version 0.2.68

Released 2021-05-27

---

## Version 0.2.67

### Critical performance fix

Released 2021-05-06

- (hail#10451) Fixed a memory leak / performance bug triggered by `hl.literal(...).contains(...)`

---

## Version 0.2.66

Released 2021-05-03

### New features

- (hail#10398) Added new method `BlockMatrix.to_ndarray`.
- (hail#10251) Added suport for haploid GT calls to VCF combiner.

---

## Version 0.2.65

Released 2021-04-14

### Default Spark Version Change

- Starting from version 0.2.65, Hail uses Spark 3.1.1 by default. This will also allow the use of all python versions >= 3.6. By building hail from source, it is still possible to use older versions of Spark.

### New features

- (hail#10290) Added `hl.nd.solve`.
- (hail#10187) Added `NDArrayNumericExpression.sum`.

### Performance improvements

- (hail#10233) Loops created with `hl.experimental.loop` will now clean up unneeded memory between iterations.

### Bug fixes

- (hail#10227) `hl.nd.qr` now supports ndarrays that have 0 rows or columns.

---

## Version 0.2.64

Released 2021-03-11

### New features
- (hail#10164) Add source_file_field parameter to hl.import_table to allow lines to be associated with their original source file.

### Bug fixes

- (hail#10182) Fixed serious memory leak in certain uses of `filter_intervals`.
- (hail#10133) Fix bug where some pipelines incorrectly infer missingness, leading to a type error.
- (hail#10134) Teach `hl.king` to treat filtered entries as missing values.
- (hail#10158) Fixes hail usage in latest versions of jupyter that rely on `asyncio`.
- (hail#10174) Fixed bad error message when incorrect return type specified with `hl.loop`.

---


## Version 0.2.63

Released 2021-03-01

- (hail#10105) Hail will now return `frozenset` and `hail.utils.frozendict` instead of normal sets and dicts.


### Bug fixes

- (hail#10035) Fix mishandling of NaN values in `hl.agg.hist`, where they were unintentionally included in the first bin.
- (hail#10007) Improve error message from hadoop_ls when file does not exist.

### Performance Improvements

- (hail#10068) Make certain array copies faster.
- (hail#10061) Improve code generation of `hl.if_else` and `hl.coalesce`.

---

## Version 0.2.62

Released 2021-02-03

### New features

- (hail#9936) Deprecated `hl.null` in favor of `hl.missing` for naming consistency.
- (hail#9973) `hl.vep` now includes a `vep_proc_id` field to aid in debugging unexpected output.
- (hail#9839) Hail now eagerly deletes temporary files produced by some BlockMatrix operations.
- (hail#9835) `hl.any` and `hl.all` now also support a single collection argument and a varargs of Boolean expressions.
- (hail#9816) `hl.pc_relate` now includes values on the diagonal of kinship, IBD-0, IBD-1, and IBD-2
- (hail#9736) Let NDArrayExpression.reshape take varargs instead of mandating a tuple.
- (hail#9766) `hl.export_vcf` now warns if INFO field names are invalid according to the VCF 4.3 spec.

### Bug fixes

- (hail#9976) Fixed `show()` representation of Hail dictionaries.

### Performance improvements

- (hail#9909) Improved performance of `hl.experimental.densify` by approximately 35%.

---

## Version 0.2.61

Released 2020-12-03

### New features

- (hail#9749) Add or_error method to SwitchBuilder (`hl.switch`)

### Bug fixes

- (hail#9775) Fixed race condition leading to invalid intermediate files in VCF combiner.
- (hail#9751) Fix bug where constructing an array of empty structs causes type error.
- (hail#9731) Fix error and incorrect behavior when using `hl.import_matrix_table` with int64 data types.

---

## Version 0.2.60

Released 2020-11-16

### New features

- (hail#9696) `hl.experimental.export_elasticsearch` will now support Elasticsearch versions 6.8 - 7.x by default.

### Bug fixes

- (hail#9641) Showing hail ndarray data now always prints in correct order.

### hailctl dataproc

- (hail#9610) Support interval fields in `hailctl dataproc describe`

---

## Version 0.2.59

Released 2020-10-22

### Datasets / Annotation DB

- (hail#9605) The Datasets API and the Annotation Database now support AWS, and users are required to specify what cloud platform they're using.

### hailctl dataproc

- (hail#9609) Fixed bug where `hailctl dataproc modify` did not correctly print corresponding `gcloud` command.

---

## Version 0.2.58

Released 2020-10-08

### New features

- (hail#9524) Hail should now be buildable using Spark 3.0.
- (hail#9549) Add `ignore_in_sample_frequency` flag to `hl.de_novo`.
- (hail#9501) Configurable cache size for `BlockMatrix.to_matrix_table_row_major` and `BlockMatrix.to_table_row_major`.
- (hail#9474) Add `ArrayExpression.first` and `ArrayExpression.last`.
- (hail#9459) Add `StringExpression.join`, an analogue to Python's `str.join`.
- (hail#9398) Hail will now throw `HailUserError`s if the `or_error` branch of a `CaseBuilder` is hit.

### Bug fixes
- (hail#9503) NDArrays can now hold arbitrary data types, though only ndarrays of primitives can be collected to Python.
- (hail#9501) Remove memory leak in `BlockMatrix.to_matrix_table_row_major` and `BlockMatrix.to_table_row_major`.
- (hail#9424) `hl.experimental.writeBlockMatrices` didn't correctly support `overwrite` flag.

### Performance improvements
- (hail#9506) `hl.agg.ndarray_sum` will now do a tree aggregation.

### hailctl dataproc
- (hail#9502) Fix hailctl dataproc modify to install dependencies of the wheel file.
- (hail#9420) Add `--debug-mode` flag to `hailctl dataproc start`. This will enable heap dumps on OOM errors.
- (hail#9520) Add support for requester pays buckets to `hailctl dataproc describe`.

### Deprecations
- (hail#9482) `ArrayExpression.head` has been deprecated in favor of `ArrayExpression.first`.

---

## Version 0.2.57

Released 2020-09-03

### New features

- (hail#9343) Implement the KING method for relationship inference as `hl.methods.king`.

---

## Version 0.2.56

Released 2020-08-31

### New features

- (hail#9308) Add hl.enumerate in favor of hl.zip_with_index, which is now deprecated.
- (hail#9278) Add `ArrayExpression.grouped`, a function that groups hail arrays into fixed size subarrays.

### Performance

- (hail#9373)(hail#9374) Decrease amount of memory used when slicing or filtering along a single BlockMatrix dimension.

### Bug fixes

- (hail#9304) Fix crash in `run_combiner` caused by inputs where VCF lines and BGZ blocks align.

### hailctl dataproc

- (hail#9263) Add support for `--expiration-time` argument to `hailctl dataproc start`.
- (hail#9263) Add support for `--no-max-idle`, `no-max-age`, `--max-age`, and `--expiration-time` to `hailctl dataproc --modify`.

---

## Version 0.2.55

Released 2020-08-19

### Performance

- (hail#9264) Table.checkpoint now uses a faster LZ4 compression scheme.

### Bug fixes

- (hail#9250) `hailctl dataproc` no longer uses deprecated `gcloud`
  flags. Consequently, users must update to a recent version of `gcloud`.
- (hail#9294) The "Python 3" kernel in notebooks in clusters started by `hailctl
  dataproc` now features the same Spark monitoring widget found in the "Hail"
  kernel. There is now no reason to use the "Hail" kernel.

### File Format

- The native file format version is now 1.5.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

---

## Version 0.2.54

Released 2020-08-07


### VCF Combiner

- (hail#9224)(hail#9237) **Breaking change**: Users are now required to pass a partitioning argument to the command-line interface or `run_combiner` method. See documentation for details.
- (hail#8963) Improved performance of VCF combiner by ~4x.


### New features

- (hail#9209) Add `hl.agg.ndarray_sum` aggregator.

### Bug fixes

- (hail#9206)(hail#9207) Improved error messages from invalid usages of Hail expressions.
- (hail#9223) Fixed error in bounds checking for NDArray slicing.

---

## Version 0.2.53

Released 2020-07-30

### Bug fixes

- (hail#9173) Use less confusing column key behavior in MT.show.
- (hail#9172) Add a missing Python dependency to Hail: google-cloud-storage.
- (hail#9170) Change Hail tree aggregate depth logic to correctly respect the
  branching factor set in `hl.init`.

---

## Version 0.2.52

Released 2020-07-29

### Bug fixes

- (hail#8944)(hail#9169) Fixed crash (error 134 or SIGSEGV) in `MatrixTable.annotate_cols`, `hl.sample_qc`, and more.

---

## Version 0.2.51

Released 2020-07-28

### Bug fixes

- (hail#9161) Fix bug that prevented concatenating ndarrays that are fields of a table.
- (hail#9152) Fix bounds in NDArray slicing.
- (hail#9161) Fix bugs calculating *row_id* in `hl.import_matrix_table`.

---

## Version 0.2.50

Released 2020-07-23

### Bug fixes

- (hail#9114) CHANGELOG: Fixed crash when using repeated calls to `hl.filter_intervals`.

### New features

- (hail#9101) Add `hl.nd.{concat, hstack, vstack}` to concatenate ndarrays.
- (hail#9105) Add `hl.nd.{eye, identity}` to create identity matrix ndarrays.
- (hail#9093) Add `hl.nd.inv` to invert ndarrays.
- (hail#9063) Add `BlockMatrix.tree_matmul` to improve matrix multiply performance with a large inner dimension.

---

## Version 0.2.49

Released 2020-07-08

### Bug fixes

- (hail#9058) Fixed memory leak affecting `Table.aggregate`, `MatrixTable.annotate_cols` aggregations, and `hl.sample_qc`.

---

## Version 0.2.48

Released 2020-07-07

### Bug fixes

- (hail#9029) Fix crash when using `hl.agg.linreg` with no aggregated data records.
- (hail#9028) Fixed memory leak affecting `Table.annotate` with scans, `hl.experimental.densify`, and `Table.group_by` / `aggregate`.
- (hail#8978) Fixed aggregation behavior of `MatrixTable.{group_rows_by, group_cols_by}` to skip filtered entries.

---

## Version 0.2.47

Released 2020-06-23

### Bug fixes

- (hail#9009) Fix memory leak when counting per-partition. This caused excessive memory use in `BlockMatrix.write_from_entry_expr`, and likely in many other places.
- (hail#9006) Fix memory leak in `hl.export_bgen`.
- (hail#9001) Fix double close error that showed up on Azure Cloud.

## Version 0.2.46

Released 2020-06-17

### Site
- (hail#8955) Natural language documentation search

### Bug fixes
- (hail#8981) Fix BlockMatrix OOM triggered by the MatrixWriteBlockMatrix WriteBlocksRDD method

---

## Version 0.2.45

Release 2020-06-15

### Bug fixes

- (hail#8948) Fix integer overflow error when reading files >2G with
  `hl.import_plink`.
- (hail#8903) Fix Python type annotations for empty collection constructors and
  `hl.shuffle`.
- (hail#8942) Refactored VCF combiner to support other GVCF schemas.
- (hail#8941) Fixed `hl.import_plink` with multiple data partitions.

### hailctl dataproc

- (hail#8946) Fix bug when a user specifies packages in `hailctl dataproc start`
  that are also dependencies of the Hail package.
- (hail#8939) Support tuples in `hailctl dataproc describe`.

---

## Version 0.2.44

Release 2020-06-06

### New Features

- (hail#8914) `hl.export_vcf` can now export tables as sites-only VCFs.
- (hail#8894) Added `hl.shuffle` function to randomly permute arrays.
- (hail#8854) Add `composable` option to parallel text export for use with `gsutil compose`.

### Bug fixes

- (hail#8883) Fix an issue related to failures in pipelines with `force_bgz=True`.

### Performance

- (hail#8887) Substantially improve the performance of `hl.experimental.import_gtf`.

---

## Version 0.2.43

Released 2020-05-28

### Bug fixes

- (hail#8867) Fix a major correctness bug ocurring when calling BlockMatrix.transpose on sparse, non-symmetric BlockMatrices.
- (hail#8876) Fixed "ChannelClosedException: null" in `{Table, MatrixTable}.write`.

---

## Version 0.2.42

Released 2020-05-27

### New Features

- (hail#8822) Add optional non-centrality parameter to `hl.pchisqtail`.
- (hail#8861) Add `contig_recoding` option to `hl.experimental.run_combiner`.

### Bug fixes

- (hail#8863) Fixes VCF combiner to successfully import GVCFs with alleles called as <NON_REF>.
- (hail#8845) Fixed issue where accessing an element of an ndarray in a call to Table.transmute would fail.
- (hail#8855) Fix crash in `filter_intervals`.

---

## Version 0.2.41

Released 2020-05-15

### Bug fixes

- (hail#8799)(hail#8786) Fix ArrayIndexOutOfBoundsException seen in pipelines that reuse a tuple value.

### hailctl dataproc

- (hail#8790) Use configured compute zone as default for `hailctl dataproc connect` and `hailctl dataproc modify`.

---

## Version 0.2.40

Released 2020-05-12

### VCF Combiner

 - (hail#8706) Add option to key by both locus and alleles for final output.

### Bug fixes

 - (hail#8729) Fix assertion error in `Table.group_by(...).aggregate(...)`
 - (hail#8708) Fix assertion error in reading tables and matrix tables with `_intervals` option.
 - (hail#8756) Fix return type of `LocusExpression.window` to use locus's reference genome instead of default RG.

---

## Version 0.2.39

Released 2020-04-29

### Bug fixes

- (hail#8615) Fix contig ordering in the CanFam3 (dog) reference genome.
- (hail#8622) Fix bug that causes inscrutable JVM Bytecode errors.
- (hail#8645) Ease unnecessarily strict assertion that caused errors when
  aggregating by key (e.g. `hl.experimental.spread`).
- (hail#8621) `hl.nd.array` now supports arrays with no elements
  (e.g. `hl.nd.array([]).reshape((0, 5))`) and, consequently, matmul with an
  inner dimension of zero.

### New features

- (hail#8571) `hl.init(skip_logging_configuration=True)` will skip configuration
  of Log4j. Users may use this to configure their own logging.
- (hail#8588) Users who manually build Python wheels will experience less
  unnecessary output when doing so.
- (hail#8572) Add `hl.parse_json` which converts a string containing JSON into a
  Hail object.

### Performance Improvements

- (hail#8535) Increase speed of `import_vcf`.
- (hail#8618) Increase speed of Jupyter Notebook file listing and Notebook
  creation when buckets contain many objects.
- (hail#8613) `hl.experimental.export_entries_by_col` stages files for improved
  reliability and performance.

### Documentation
- (hail#8619) Improve installation documentation to suggest better performing
  LAPACK and BLAS libraries.
- (hail#8647) Clarify that a LAPACK or BLAS library is a *requirement* for a
  complete Hail installation.
- (hail#8654) Add link to document describing the creation of a Microsoft Azure
  HDInsight Hail cluster.

---

## Version 0.2.38

Released 2020-04-21

### Critical Linreg Aggregator Correctness Bug

- (hail#8575) Fixed a correctness bug in the linear regression aggregator. This was introduced in version 0.2.29.
See https://discuss.hail.is/t/possible-incorrect-linreg-aggregator-results-in-0-2-29-0-2-37/1375 for more details.

### Performance improvements
- (hail#8558) Make `hl.experimental.export_entries_by_col` more fault tolerant.

----

## Version 0.2.37

Released 2020-04-14

### Bug fixes

- (hail#8487) Fix incorrect handling of badly formatted data for `hl.gp_dosage`.
- (hail#8497) Fix handling of missingness for `hl.hamming`.
- (hail#8537) Fix compile-time errror.
- (hail#8539) Fix compiler error in `Table.multi_way_zip_join`.
- (hail#8488) Fix `hl.agg.call_stats` to appropriately throw an error for badly-formatted calls.

### New features

- (hail#8327) Attempting to write to the same file being read from in a pipeline will now throw an error instead of corrupting data.

---

## Version 0.2.36

Released 2020-04-06

### Critical Memory Management Bug Fix

- (hail#8463) Reverted a change (separate to the bug in 0.2.34) that led to a memory leak in version 0.2.35.

### Bug fixes
- (hail#8371) Fix runtime error in joins leading to "Cannot set required field missing" error message.
- (hail#8436) Fix compiler bug leading to possibly-invalid generated code.

---

## Version 0.2.35

Released 2020-04-02

### Critical Memory Management Bug Fix

- (hail#8412) Fixed a serious per-partition memory leak that causes certain pipelines to run out of memory unexpectedly. Please update from 0.2.34.

### New features

- (hail#8404) Added "CanFam3" (a reference genome for dogs) as a bundled reference genome.

### Bug fixes

- (hail#8420) Fixed a bug where `hl.binom_test`'s `"lower"` and `"upper"` alternative options were reversed.
- (hail#8377) Fixed "inconsistent agg or scan environments" error.
- (hail#8322) Fixed bug where `aggregate_rows` did not interact with `hl.agg.array_agg` correctly.

### Performance Improvements

- (hail#8413) Improves internal region memory management, decreasing JVM overhead.
- (hail#8383) Significantly improve GVCF import speed.
- (hail#8358) Fixed memory leak in `hl.experimental.export_entries_by_col`.
- (hail#8326) Codegen infrastructure improvement resulting in ~3% overall speedup.

### hailctl dataproc

- (hail#8399) Enable spark speculation by default.
- (hail#8340) Add new Australia region to `--vep`.
- (hail#8347) Support all GCP machine types as potential master machines.

---

## Version 0.2.34

Released 2020-03-12

### New features

- (hail#8233) `StringExpression.matches` can now take a hail `StringExpression`, as opposed to only regular python strings.
- (hail#8198) Improved matrix multiplication interoperation between hail `NDArrayExpression` and numpy.

### Bug fixes

- (hail#8279) Fix a bug where `hl.agg.approx_cdf` failed inside of a `group_cols_by`.
- (hail#8275) Fix bad error message coming from `mt.make_table()` when keys are missing.
- (hail#8274) Fix memory leak in `hl.export_bgen`.
- (hail#8273) Fix segfault caused by `hl.agg.downsample` inside of an `array_agg` or `group_by`.

### hailctl dataproc

- (hail#8253) `hailctl dataproc` now supports new flags `--requester-pays-allow-all` and `--requester-pays-allow-buckets`. This will configure your hail installation to be able to read from requester pays buckets. The charges for reading from these buckets will be billed to the project that the cluster is created in.
- (hail#8268) The data sources for VEP have been moved to `gs://hail-us-vep`, `gs://hail-eu-vep`, and `gs://hail-uk-vep`, which are requester-pays buckets in Google Cloud. `hailctl dataproc` will automatically infer which of these buckets you should pull data from based on the region your cluster is spun up in. If you are in none of those regions, please contact us on discuss.hail.is.

### File Format

- The native file format version is now 1.4.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

---


## Version 0.2.33

Released 2020-02-27

### New features

- (hail#8173) Added new method `hl.zeros`.

### Bug fixes

- (hail#8153) Fixed complier bug causing `MatchError` in `import_bgen`.
- (hail#8123) Fixed an issue with multiple Python HailContexts running on the same cluster.
- (hail#8150) Fixed an issue where output from VEP about failures was not reported in error message.
- (hail#8152) Fixed an issue where the row count of a MatrixTable coming from `import_matrix_table` was incorrect.
- (hail#8175) Fixed a bug where `persist` did not actually do anything.

### `hailctl dataproc`

- (hail#8079) Using `connect` to open the jupyter notebook browser will no longer crash if your project contains requester-pays buckets.

---

## Version 0.2.32

Released 2020-02-07

### Critical performance regression fix

- (hail#7989) Fixed performance regression leading to a large slowdown when `hl.variant_qc` was run after filtering columns.

### Performance

- (hail#7962) Improved performance of `hl.pc_relate`.
- (hail#8032) Drastically improve performance of pipelines calling `hl.variant_qc` and `hl.sample_qc` iteratively.
- (hail#8037) Improve performance of NDArray matrix multiply by using native linear algebra libraries.

### Bug fixes

- (hail#7976) Fixed divide-by-zero error in `hl.concordance` with no overlapping rows or cols.
- (hail#7965) Fixed optimizer error leading to crashes caused by `MatrixTable.union_rows`.
- (hail#8035) Fix compiler bug in `Table.multi_way_zip_join`.
- (hail#8021) Fix bug in computing shape after `BlockMatrix.filter`.
- (hail#7986) Fix error in NDArray matrix/vector multiply.

### New features

- (hail#8007) Add `hl.nd.diagonal` function.

### Cheat sheets

 - (hail#7940) Added cheat sheet for MatrixTables.
 - (hail#7963) Improved Table sheet sheet.

---

## Version 0.2.31

Released 2020-01-22

### New features

- (hail#7787) Added transition/transversion information to `hl.summarize_variants`.
- (hail#7792) Add Python stack trace to array index out of bounds errors in Hail pipelines.
- (hail#7832) Add `spark_conf` argument to `hl.init`, permitting configuration of Spark runtime for a Hail session.
- (hail#7823) Added datetime functions `hl.experimental.strptime` and `hl.experimental.strftime`.
- (hail#7888) Added `hl.nd.array` constructor from nested standard arrays.

### File size

- (hail#7923) Fixed compression problem since 0.2.23 resulting in larger-than-expected matrix table files for datasets with few entry fields (e.g. GT-only datasets).

### Performance

- (hail#7867) Fix performance regression leading to extra scans of data when `order_by` and `key_by` appeared close together.
- (hail#7901) Fix performance regression leading to extra scans of data when `group_by/aggregate` and `key_by` appeared close together.
- (hail#7830) Improve performance of array arithmetic.

### Bug fixes

- (hail#7922) Fix still-not-well-understood serialization error about ApproxCDFCombiner.
- (hail#7906) Fix optimizer error by relaxing unnecessary assertion.
- (hail#7788) Fix possible memory leak in `ht.tail` and `ht.head`.
- (hail#7796) Fix bug in ingesting numpy arrays not in row-major orientation.

---

## Version 0.2.30

Released 2019-12-20

### Performance
- (hail#7771) Fixed extreme performance regression in scans.
- (hail#7764) Fixed `mt.entry_field.take` performance regression.

### New features
- (hail#7614) Added experimental support for loops with `hl.experimental.loop`.

### Miscellaneous
- (hail#7745) Changed `export_vcf` to only use scientific notation when necessary.

---

## Version 0.2.29

Released 2019-12-17

### Bug fixes
- (hail#7229) Fixed `hl.maximal_independent_set` tie breaker functionality.
- (hail#7732) Fixed incompatibility with old files leading to incorrect data read when filtering intervals after `read_matrix_table`.
- (hail#7642) Fixed crash when constant-folding functions that throw errors.
- (hail#7611) Fixed `hl.hadoop_ls` to handle glob patterns correctly.
- (hail#7653) Fixed crash in `ld_prune` by unfiltering missing GTs.

### Performance improvements
- (hail#7719) Generate more efficient IR for `Table.flatten`.
- (hail#7740) Method wrapping large let bindings to keep method size down.

### New features
- (hail#7686) Added `comment` argument to `import_matrix_table`, allowing lines with certain prefixes to be ignored.
- (hail#7688) Added experimental support for `NDArrayExpression`s in new `hl.nd` module.
- (hail#7608) `hl.grep` now has a `show` argument that allows users to either print the results (default) or return a dictionary of the results.

### `hailctl dataproc`
- (hail#7717) Throw error when mispelling arguments instead of silently quitting.

---

## Version 0.2.28

Released 2019-11-22

### Critical correctness bug fix
- (hail#7588) Fixes a bug where filtering old matrix tables in newer versions of hail did not work as expected. Please update from 0.2.27.

### Bug fixes
- (hail#7571) Don't set GQ to missing if PL is missing in `split_multi_hts`.
- (hail#7577) Fixed an optimizer bug.

### New Features
- (hail#7561) Added `hl.plot.visualize_missingness()` to plot missingness patterns for MatrixTables.
- (hail#7575) Added `hl.version()` to quickly check hail version.

### `hailctl dataproc`
- (hail#7586) `hailctl dataproc` now supports `--gcloud_configuration` option.

### Documentation
- (hail#7570) Hail has a cheatsheet for Tables now.

---

## Version 0.2.27

Released 2019-11-15

### New Features

- (hail#7379) Add `delimiter` argument to `hl.import_matrix_table`
- (hail#7389) Add `force` and `force_bgz` arguments to `hl.experimental.import_gtf`
- (hail#7386)(hail#7394) Add `{Table, MatrixTable}.tail`.
- (hail#7467) Added `hl.if_else` as an alias for `hl.cond`; deprecated `hl.cond`.
- (hail#7453) Add `hl.parse_int{32, 64}` and `hl.parse_float{32, 64}`, which can parse strings to numbers and return missing on failure.
- (hail#7475) Add `row_join_type` argument to `MatrixTable.union_cols` to support outer joins on rows.

### Bug fixes

- (hail#7479)(hail#7368)(hail#7402) Fix optimizer bugs.
- (hail#7506) Updated to latest htsjdk to resolve VCF parsing problems.

### `hailctl dataproc`

- (hail#7460) The Spark monitor widget now automatically collapses after a job completes.

---

## Version 0.2.26

Released 2019-10-24

### New Features
- (hail#7325) Add `string.reverse` function.
- (hail#7328) Add `string.translate` function.
- (hail#7344) Add `hl.reverse_complement` function.
- (hail#7306) Teach the VCF combiner to handle allele specific (`AS_*`) fields.
- (hail#7346) Add `hl.agg.approx_median` function.

### Bug Fixes
- (hail#7361) Fix `AD` calculation in `sparse_split_multi`.

### Performance Improvements
- (hail#7355) Improve performance of IR copying.

### File Format

- The native file format version is now 1.3.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

## Version 0.2.25

Released 2019-10-14

### New features
- (hail#7240) Add interactive schema widget to `{MatrixTable, Table}.describe`. Use this by passing the argument `widget=True`.
- (hail#7250) `{Table, MatrixTable, Expression}.summarize()` now summarizes elements of collections (arrays, sets, dicts).
- (hail#7271) Improve `hl.plot.qq` by increasing point size, adding the unscaled p-value to hover data, and printing lambda-GC on the plot.
- (hail#7280) Add HTML output for `{Table, MatrixTable, Expression}.summarize()`.
- (hail#7294) Add HTML output for `hl.summarize_variants()`.

### Bug fixes
- (hail#7200) Fix VCF parsing with missingness inside arrays of floating-point values in the FORMAT field.
- (hail#7219) Fix crash due to invalid optimizer rule.

### Performance improvements
- (hail#7187) Dramatically improve performance of chained `BlockMatrix` multiplies without checkpoints in between.
- (hail#7195)(hail#7194) Improve performance of `group[_rows]_by` / `aggregate`.
- (hail#7201) Permit code generation of larger aggregation pipelines.

### File Format

- The native file format version is now 1.2.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

---

## Version 0.2.24

Released 2019-10-03

### `hailctl dataproc`
- (hail#7185) Resolve issue in dependencies that led to a Jupyter update breaking cluster creation.

### New features
- (hail#7071) Add `permit_shuffle` flag to `hl.{split_multi, split_multi_hts}` to allow processing of datasets with both multiallelics and duplciate loci.
- (hail#7121) Add `hl.contig_length` function.
- (hail#7130) Add `window` method on `LocusExpression`, which creates an interval around a locus.
- (hail#7172) Permit `hl.init(sc=sc)` with pip-installed packages, given the right configuration options.

### Bug fixes
- (hail#7070) Fix unintentionally strict type error in `MatrixTable.union_rows`.
- (hail#7170) Fix issues created downstream of `BlockMatrix.T`.
- (hail#7146) Fix bad handling of edge cases in `BlockMatrix.filter`.
- (hail#7182) Fix problem parsing VCFs where lines end in an INFO field of type flag.

---

## Version 0.2.23

Released 2019-09-23

### `hailctl dataproc`
- (hail#7087) Added back progress bar to notebooks, with links to the correct Spark UI url.
- (hail#7104) Increased disk requested when using `--vep` to address the "colony collapse" cluster error mode.

### Bug fixes
- (hail#7066) Fixed generated code when methods from multiple reference genomes appear together.
- (hail#7077) Fixed crash in `hl.agg.group_by`.

### New features
- (hail#7009) Introduced analysis pass in Python that mostly obviates the `hl.bind` and `hl.rbind` operators; idiomatic Python that generates Hail expressions will perform much better.
- (hail#7076) Improved memory management in generated code, add additional log statements about allocated memory to improve debugging.
- (hail#7085) Warn only once about schema mismatches during JSON import (used in VEP, Nirvana, and sometimes `import_table`.
- (hail#7106) `hl.agg.call_stats` can now accept a number of alleles for its `alleles` parameter, useful when dealing with biallelic calls without the alleles array at hand.

### Performance
- (hail#7086) Improved performance of JSON import.
- (hail#6981) Improved performance of Hail min/max/mean operators. Improved performance of `split_multi_hts` by an additional 33%.
- (hail#7082)(hail#7096)(hail#7098) Improved performance of large pipelines involving many `annotate` calls.

---

## Version 0.2.22

Released 2019-09-12

### New features
- (hail#7013) Added `contig_recoding` to `import_bed` and `import_locus_intervals`.

### Performance
- (hail#6969) Improved performance of `hl.agg.mean`, `hl.agg.stats`, and `hl.agg.corr`.
- (hail#6987) Improved performance of `import_matrix_table`.
- (hail#7033)(hail#7049) Various improvements leading to overall 10-15%
  improvement.

### `hailctl dataproc`
- (hail#7003) Pass through extra arguments for `hailctl dataproc list` and
  `hailctl dataproc stop`.

---

## Version 0.2.21

Released 2019-09-03

### Bug fixes
- (hail#6945) Fixed `expand_types` to preserve ordering by key, also affects
    `to_pandas` and `to_spark`.
- (hail#6958) Fixed stack overflow errors when counting the result of a `Table.union`.

### New features
- (hail#6856) Teach `hl.agg.counter` to weigh each value differently.
- (hail#6903) Teach `hl.range` to treat a single argument as `0..N`.
- (hail#6903) Teach `BlockMatrix` how to `checkpoint`.

### Performance
- (hail#6895) Improved performance of `hl.import_bgen(...).count()`.
- (hail#6948) Fixed performance bug in `BlockMatrix` filtering functions.
- (hail#6943) Improved scaling of `Table.union`.
- (hail#6980) Reduced compute time for `split_multi_hts` by as much as 40%.

### `hailctl dataproc`
- (hail#6904) Added `--dry-run` option to `submit`.
- (hail#6951) Fixed `--max-idle` and `--max-age` arguments to `start`.
- (hail#6919) Added `--update-hail-version` to `modify`.

---

## Version 0.2.20

Released 2019-08-19

### Critical memory management fix

- (hail#6824) Fixed memory management inside `annotate_cols` with
  aggregations. This was causing memory leaks and segfaults.

### Bug fixes
- (hail#6769) Fixed non-functional `hl.lambda_gc` method.
- (hail#6847) Fixed bug in handling of NaN in `hl.agg.min` and `hl.agg.max`.
  These will now properly ignore NaN (the intended semantics). Note that
  `hl.min` and `hl.max` propagate NaN; use `hl.nanmin` and  `hl.nanmax`
  to ignore NaN.

### New features
- (hail#6847) Added `hl.nanmin` and `hl.nanmax` functions.

-----

## Version 0.2.19

Released 2019-08-01

### Critical performance bug fix

- (hail#6629) Fixed a critical performance bug introduced in (hail#6266).
  This bug led to long hang times when reading in Hail tables and matrix
  tables **written in version 0.2.18**.

### Bug fixes
- (hail#6757) Fixed correctness bug in optimizations applied to the
  combination of `Table.order_by` with `hl.desc` arguments and `show()`,
  leading to tables sorted in ascending, not descending order.
- (hail#6770) Fixed assertion error caused by `Table.expand_types()`,
  which was used by `Table.to_spark` and `Table.to_pandas`.

### Performance Improvements

- (hail#6666) Slightly improve performance of `hl.pca` and
  `hl.hwe_normalized_pca`.
- (hail#6669) Improve performance of `hl.split_multi` and
  `hl.split_multi_hts`.
- (hail#6644) Optimize core code generation primitives, leading to
  across-the-board performance improvements.
- (hail#6775) Fixed a major performance problem related to reading block
  matrices.

### `hailctl dataproc`

- (hail#6760) Fixed the address pointed at by `ui`  in `connect`, after
  Google changed proxy settings that rendered the UI URL incorrect. Also
  added new address `hist/spark-history`.

-----

## Version 0.2.18

Released 2019-07-12

### Critical performance bug fix

- (hail#6605) Resolved code generation issue leading a performance
  regression of 1-3 orders of magnitude in Hail pipelines using
  constant strings or literals. This includes almost every pipeline!
  **This issue has exists in versions 0.2.15, 0.2.16, and 0.2.17, and
  any users on those versions should update as soon as possible.**

### Bug fixes

- (hail#6598) Fixed code generated by `MatrixTable.unfilter_entries` to
  improve performance. This will slightly improve the performance of
  `hwe_normalized_pca` and relatedness computation methods, which use
  `unfilter_entries` internally.

-----

## Version 0.2.17

Released 2019-07-10

### New features

- (hail#6349) Added `compression` parameter to `export_block_matrices`, which can
  be `'gz'` or `'bgz'`.
- (hail#6405) When a matrix table has string column-keys, `matrixtable.show` uses
  the column key as the column name.
- (hail#6345) Added an improved scan implementation, which reduces the memory
  load on master.
- (hail#6462) Added `export_bgen` method.
- (hail#6473) Improved performance of `hl.agg.array_sum` by about 50%.
- (hail#6498) Added method `hl.lambda_gc` to calculate the genomic control inflation factor.
- (hail#6456) Dramatically improved performance of pipelines containing long chains of calls to
  `Table.annotate`, or `MatrixTable` equivalents.
- (hail#6506) Improved the performance of the generated code for the `Table.annotate(**thing)`
  pattern.

### Bug fixes

- (hail#6404) Added `n_rows` and `n_cols` parameters to `Expression.show` for
  consistency with other `show` methods.
- (hail#6408)(hail#6419) Fixed an issue where the `filter_intervals` optimization
  could make scans return incorrect results.
- (hail#6459)(hail#6458) Fixed rare correctness bug in the `filter_intervals`
  optimization which could result too many rows being kept.
- (hail#6496) Fixed html output of `show` methods to truncate long field
  contents.
- (hail#6478) Fixed the broken documentation for the experimental `approx_cdf`
  and `approx_quantiles` aggregators.
- (hail#6504) Fix `Table.show` collecting data twice while running in Jupyter notebooks.
- (hail#6571) Fixed the message printed in `hl.concordance` to print the number of overlapping
  samples, not the full list of overlapping sample IDs.
- (hail#6583) Fixed `hl.plot.manhattan` for non-default reference genomes.

### Experimental

- (hail#6488) Exposed `table.multi_way_zip_join`. This takes a list of tables of
  identical types, and zips them together into one table.

### File Format

- The native file format version is now 1.1.0. Older versions of Hail will not
  be able to read tables or matrix tables written by this version of Hail.

-----

## Version 0.2.16

Released 2019-06-19

### `hailctl`

- (hail#6357) Accommodated Google Dataproc bug causing cluster creation failures.

### Bug fixes

- (hail#6378) Fixed problem in how `entry_float_type` was being handled in `import_vcf`.

-----

## Version 0.2.15

Released 2019-06-14

After some infrastructural changes to our development process, we should be
getting back to frequent releases.

### `hailctl`

Starting in 0.2.15, `pip` installations of Hail come bundled with a command-
line tool, `hailctl`. This tool subsumes the functionality of `cloudtools`,
which is now deprecated. See the
[release thread on the forum](https://discuss.hail.is/t/new-command-line-utility-hailctl/981)
for more information.

### New features

- (hail#5932)(hail#6115) `hl.import_bed` abd `hl.import_locus_intervals` now
  accept keyword arguments to pass through to `hl.import_table`, which is used
  internally. This permits parameters like `min_partitions` to be set.
- (hail#5980) Added `log` option to `hl.plot.histogram2d`.
- (hail#5937) Added `all_matches` parameter to `Table.index` and
  `MatrixTable.index_{rows, cols, entries}`, which produces an array of all
  rows in the indexed object matching the index key. This makes it possible to,
  for example, annotate all intervals overlapping a locus.
- (hail#5913) Added functionality that makes arrays of structs easier to work
  with.
- (hail#6089) Added HTML output to `Expression.show` when running in a notebook.
- (hail#6172) `hl.split_multi_hts` now uses the original `GQ` value if the `PL`
  is missing.
- (hail#6123) Added `hl.binary_search` to search sorted numeric arrays.
- (hail#6224) Moved implementation of `hl.concordance` from backend to Python.
  Performance directly from `read()` is slightly worse, but inside larger
  pipelines this function will be optimized much better than before, and it
  will benefit improvements to general infrastructure.
- (hail#6214) Updated Hail Python dependencies.
- (hail#5979) Added optimizer pass to rewrite filter expressions on keys as
  interval filters where possible, leading to massive speedups for point queries.
  See the [blog post](https://discuss.hail.is/t/new-optimizer-pass-that-extracts-point-queries-and-interval-filters/979)
  for examples.

### Bug fixes

- (hail#5895) Fixed crash caused by `-0.0` floating-point values in `hl.agg.hist`.
- (hail#6013) Turned off feature in HTSJDK that caused crashes in `hl.import_vcf`
  due to header fields being overwritten with different types, if the field had
  a different type than the type in the VCF 4.2 spec.
- (hail#6117) Fixed problem causing `Table.flatten()` to be quadratic in the size
  of the schema.
- (hail#6228)(hail#5993) Fixed `MatrixTable.union_rows()` to join distinct keys
  on the right, preventing an unintentional cartesian product.
- (hail#6235) Fixed an issue related to aggregation inside `MatrixTable.filter_cols`.
- (hail#6226) Restored lost behavior where `Table.show(x < 0)` shows the entire table.
- (hail#6267) Fixed cryptic crashes related to `hl.split_multi` and `MatrixTable.entries()`
  with duplicate row keys.

-----

## Version 0.2.14

Released 2019-04-24

A back-incompatible patch update to PySpark, 2.4.2, has broken fresh pip
installs of Hail 0.2.13. To fix this, either *downgrade* PySpark to 2.4.1 or
upgrade to the latest version of Hail.

### New features

- (hail#5915) Added `hl.cite_hail` and `hl.cite_hail_bibtex` functions to
  generate appropriate citations.
- (hail#5872) Fixed `hl.init` when the `idempotent` parameter is `True`.

-----

## Version 0.2.13

Released 2019-04-18

Hail is now using Spark 2.4.x by default. If you build hail from source, you
will need to acquire this version of Spark and update your build invocations
accordingly.

### New features

- (hail#5828) Remove dependency on htsjdk for VCF INFO parsing, enabling
  faster import of some VCFs.
- (hail#5860) Improve performance of some column annotation pipelines.
- (hail#5858) Add `unify` option to `Table.union` which allows unification of
  tables with different fields or field orderings.
- (hail#5799) `mt.entries()` is four times faster.
- (hail#5756) Hail now uses Spark 2.4.x by default.
- (hail#5677) `MatrixTable` now also supports `show`.
- (hail#5793)(hail#5701) Add `array.index(x)` which find the first index of
  `array` whose value is equal to `x`.
- (hail#5790) Add `array.head()` which returns the first element of the array,
  or missing if the array is empty.
- (hail#5690) Improve performance of `ld_matrix`.
- (hail#5743) `mt.compute_entry_filter_stats` computes statistics about the number
  of filtered entries in a matrix table.
- (hail#5758) failure to parse an interval will now produce a much more detailed
  error message.
- (hail#5723) `hl.import_matrix_table` can now import a matrix table with no
  columns.
- (hail#5724) `hl.rand_norm2d` samples from a two dimensional random normal.

### Bug fixes

- (hail#5885) Fix `Table.to_spark` in the presence of fields of tuples.
- (hail#5882)(hail#5886) Fix `BlockMatrix` conversion methods to correctly
  handle filtered entries.
- (hail#5884)(hail#4874) Fix longstanding crash when reading Hail data files
  under certain conditions.
- (hail#5855)(hail#5786) Fix `hl.mendel_errors` incorrectly reporting children counts in
  the presence of entry filtering.
- (hail#5830)(hail#5835) Fix Nirvana support
- (hail#5773) Fix `hl.sample_qc` to use correct number of total rows when
  calculating call rate.
- (hail#5763)(hail#5764) Fix `hl.agg.array_agg` to work inside
  `mt.annotate_rows` and similar functions.
- (hail#5770) Hail now uses the correct unicode string encoding which resolves a
  number of issues when a Table or MatrixTable has a key field containing
  unicode characters.
- (hail#5692) When `keyed` is `True`, `hl.maximal_independent_set` now does not
  produce duplicates.
- (hail#5725) Docs now consistently refer to `hl.agg` not `agg`.
- (hail#5730)(hail#5782) Taught `import_bgen` to optimize its `variants` argument.

### Experimental

- (hail#5732) The `hl.agg.approx_quantiles` aggregate computes an approximation
  of the quantiles of an expression.
- (hail#5693)(hail#5396) `Table._multi_way_zip_join` now correctly handles keys
  that have been truncated.

-----

## Version 0.2.12

Released 2019-03-28

### New features

- (hail#5614) Add support for multiple missing values in `hl.import_table`.
- (hail#5666) Produce HTML table output for `Table.show()` when running in Jupyter notebook.

### Bug fixes

- (hail#5603)(hail#5697) Fixed issue where `min_partitions` on `hl.import_table` was non-functional.
- (hail#5611) Fix `hl.nirvana` crash.

### Experimental

- (hail#5524) Add `summarize` functions to Table, MatrixTable, and Expression.
- (hail#5570) Add `hl.agg.approx_cdf` aggregator for approximate density calculation.
- (hail#5571) Add `log` parameter to `hl.plot.histogram`.
- (hail#5601) Add `hl.plot.joint_plot`, extend functionality of `hl.plot.scatter`.
- (hail#5608) Add LD score simulation framework.
- (hail#5628) Add `hl.experimental.full_outer_join_mt` for full outer joins on `MatrixTable`s.

-----

## Version 0.2.11

Released 2019-03-06

### New features

- (hail#5374) Add default arguments to `hl.add_sequence` for running on GCP.
- (hail#5481) Added `sample_cols` method to `MatrixTable`.
- (hail#5501) Exposed `MatrixTable.unfilter_entries`. See `filter_entries` documentation for more information.
- (hail#5480) Added `n_cols` argument to `MatrixTable.head`.
- (hail#5529) Added `Table.{semi_join, anti_join}` and `MatrixTable.{semi_join_rows, semi_join_cols, anti_join_rows, anti_join_cols}`.
- (hail#5528) Added `{MatrixTable, Table}.checkpoint` methods as wrappers around `write` / `read_{matrix_table, table}`.

### Bug fixes

- (hail#5416) Resolved issue wherein VEP and certain regressions were recomputed on each use, rather than once.
- (hail#5419) Resolved issue with `import_vcf` `force_bgz` and file size checks.
- (hail#5427) Resolved issue with `Table.show` and dictionary field types.
- (hail#5468) Resolved ordering problem with `Expression.show` on key fields that are not the first key.
- (hail#5492) Fixed `hl.agg.collect` crashing when collecting `float32` values.
- (hail#5525) Fixed `hl.trio_matrix` crashing when `complete_trios` is `False`.

-----

## Version 0.2.10

Released 2019-02-15

### New features

- (hail#5272) Added a new 'delimiter' option to Table.export.
- (hail#5251) Add utility aliases to `hl.plot` for `output_notebook` and `show`.
- (hail#5249) Add `histogram2d` function to `hl.plot` module.
- (hail#5247) Expose `MatrixTable.localize_entries` method for converting to a Table with an entries array.
- (hail#5300) Add new `filter` and `find_replace` arguments to `hl.import_table` and `hl.import_vcf` to apply regex and substitutions to text input.

### Performance improvements

- (hail#5298) Reduce size of exported VCF files by exporting missing genotypes without trailing fields.

### Bug fixes

- (hail#5306) Fix `ReferenceGenome.add_sequence` causing a crash.
- (hail#5268) Fix `Table.export` writing a file called 'None' in the current directory.
- (hail#5265) Fix `hl.get_reference` raising an exception when called before `hl.init()`.
- (hail#5250) Fix crash in `pc_relate` when called on a MatrixTable field other than 'GT'.
- (hail#5278) Fix crash in `Table.order_by` when sorting by fields whose names are not valid Python identifiers.
- (hail#5294) Fix crash in `hl.trio_matrix` when sample IDs are missing.
- (hail#5295) Fix crash in `Table.index` related to key field incompatibilities.

-----

## Version 0.2.9

Released 2019-01-30

### New features

 - (hail#5149) Added bitwise transformation functions: `hl.bit_{and, or, xor, not, lshift, rshift}`.
 - (hail#5154) Added `hl.rbind` function, which is similar to `hl.bind` but expects a function as the last argument instead of the first.

### Performance improvements

 - (hail#5107) Hail's Python interface generates tighter intermediate code, which should result in moderate performance improvements in many pipelines.
 - (hail#5172) Fix unintentional performance deoptimization related to `Table.show` introduced in 0.2.8.
 - (hail#5078) Improve performance of `hl.ld_prune` by up to 30x.

### Bug fixes

 - (hail#5144) Fix crash caused by `hl.index_bgen` (since 0.2.7)
 - (hail#5177) Fix bug causing `Table.repartition(n, shuffle=True)` to fail to increase partitioning for unkeyed tables.
 - (hail#5173) Fix bug causing `Table.show` to throw an error when the table is empty (since 0.2.8).
 - (hail#5210) Fix bug causing `Table.show` to always print types, regardless of `types` argument (since 0.2.8).
 - (hail#5211) Fix bug causing `MatrixTable.make_table` to unintentionally discard non-key row fields (since 0.2.8).

-----

## Version 0.2.8

Released 2019-01-15

### New features

 - (hail#5072) Added multi-phenotype option to `hl.logistic_regression_rows`
 - (hail#5077) Added support for importing VCF floating-point FORMAT fields as `float32` as well as `float64`.

### Performance improvements

 - (hail#5068) Improved optimization of `MatrixTable.count_cols`.
 - (hail#5131) Fixed performance bug related to `hl.literal` on large values with missingness

### Bug fixes

 - (hail#5088) Fixed name separator in `MatrixTable.make_table`.
 - (hail#5104) Fixed optimizer bug related to experimental functionality.
 - (hail#5122) Fixed error constructing `Table` or `MatrixTable` objects with fields with certain character patterns like `$`.

-----

## Version 0.2.7

Released 2019-01-03

### New features

 - (hail#5046)(experimental) Added option to BlockMatrix.export_rectangles to export as NumPy-compatible binary.

### Performance improvements

 - (hail#5050) Short-circuit iteration in `logistic_regression_rows` and `poisson_regression_rows` if NaNs appear.

-----

## Version 0.2.6

Released 2018-12-17

### New features

 - (hail#4962) Expanded comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) to support expressions of every type.
 - (hail#4927) Expanded functionality of `Table.order_by` to support ordering by arbitrary expressions, instead of just top-level fields.
 - (hail#4926) Expanded default GRCh38 contig recoding behavior in `import_plink`.

### Performance improvements

 - (hail#4952) Resolved lingering issues related to (hail#4909).

### Bug fixes

 - (hail#4941) Fixed variable scoping error in regression methods.
 - (hail#4857) Fixed bug in maximal_independent_set appearing when nodes were named something other than `i` and `j`.
 - (hail#4932) Fixed possible error in `export_plink` related to tolerance of writer process failure.
 - (hail#4920) Fixed bad error message in `Table.order_by`.

-----

## Version 0.2.5

Released 2018-12-07

### New features

 - (hail#4845) The [or_error](https://hail.is/docs/0.2/functions/core.html#hail.expr.builders.CaseBuilder.or_error) method in `hl.case` and `hl.switch` statements now takes a string expression rather than a string literal, allowing more informative messages for errors and assertions.
 - (hail#4865) We use this new `or_error` functionality in methods that require biallelic variants to include an offending variant in the error message.
 - (hail#4820) Added [hl.reversed](https://hail.is/docs/0.2/functions/collections.html#hail.expr.functions.reversed) for reversing arrays and strings.
 - (hail#4895) Added `include_strand` option to the [hl.liftover](https://hail.is/docs/0.2/functions/genetics.html#hail.expr.functions.liftover) function.


### Performance improvements

 - (hail#4907)(hail#4911) Addressed one aspect of bad scaling in enormous literal values (triggered by a list of 300,000 sample IDs) related to logging.
 - (hail#4909)(hail#4914) Fixed a check in Table/MatrixTable initialization that scaled O(n^2) with the total number of fields.

### Bug fixes

 - (hail#4754)(hail#4799) Fixed optimizer assertion errors related to certain types of pipelines using ``group_rows_by``.
 - (hail#4888) Fixed assertion error in BlockMatrix.sum.
 - (hail#4871) Fixed possible error in locally sorting nested collections.
 - (hail#4889) Fixed break in compatibility with extremely old MatrixTable/Table files.
 - (hail#4527)(hail#4761) Fixed optimizer assertion error sometimes encountered with ``hl.split_multi[_hts]``.

-----

## Version 0.2.4: Beginning of history!

We didn't start manually curating information about user-facing changes until version 0.2.4.

The full commit history is available [here](https://github.com/hail-is/hail/commits/main).

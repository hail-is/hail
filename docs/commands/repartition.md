<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

Repartition is used to change Hail distribution behavior.  This command is best used to reduce the number of partitions after filtering out large amounts of data before `write` or `pca` steps.  It can also be used to increase the number of partitions before large joins where the partition size will increase.

The `--no-shuffle` option uses a different model which does not use disk for intermediates, and consequently **cannot** be used to increase the number of partitions.  The `--no-shuffle` option will increase performance in some edge cases, but often will perform worse than the default mode.

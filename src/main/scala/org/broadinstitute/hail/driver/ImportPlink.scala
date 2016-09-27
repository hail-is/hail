package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.plink.{FamFileConfig, PlinkLoader}
import org.kohsuke.args4j.{Option => Args4jOption}

object ImportPlink extends Command {
  def name = "importplink"

  def description = "Load PLINK binary file (.bed, .bim, .fam) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "--bfile", usage = "Plink Binary file root name", forbids = Array("--bed", "--bim", "--fam"))
    var bfile: String = _

    @Args4jOption(name = "--bed", usage = "Plink .bed file", forbids = Array("--bfile"), depends = Array("--bim", "--fam"))
    var bed: String = _

    @Args4jOption(name = "--bim", usage = "Plink .bim file", forbids = Array("--bfile"), depends = Array("--bed", "--fam"))
    var bim: String = _

    @Args4jOption(name = "--fam", usage = "Plink .fam file", forbids = Array("--bfile"), depends = Array("--bim", "--bed"))
    var fam: String = _

    @Args4jOption(name = "--no-compress", usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: java.lang.Integer = _

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = ".fam file field delimiter regex")
    var famDelimiter: String = "\\\\s+"

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = ".fam identifier to be treated as missing (for case-control, in addition to `0', `-9', and non-numeric)")
    var famMissing: String = "NA"

    @Args4jOption(required = false, name = "-q", aliases = Array("--quantpheno"),
      usage = ".fam file quantitative phenotype flag")
    var famIsQuantitative: Boolean = false
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val hasBFile = options.bfile != null
    val hasBedBimFam = options.bed != null || options.bim != null || options.fam != null
    if ((!hasBFile && !hasBedBimFam) || (hasBFile && hasBedBimFam))
      fatal("invalid input: require either --bed/--bim/--fam arguments or --bfile argument")

    val nPartitionOption = Option(options.nPartitions).map(_.toInt)
    val ffConfig = FamFileConfig(options.famIsQuantitative, options.famDelimiter, options.famMissing)

    state.hadoopConf.setBoolean("compressGS", !options.noCompress)

    if (options.bfile != null)
      state.copy(vds = PlinkLoader(options.bfile + ".bed", options.bfile + ".bim", options.bfile + ".fam",
        ffConfig, state.sc, nPartitionOption))
    else
      state.copy(vds = PlinkLoader(options.bed, options.bim, options.fam,
        ffConfig, state.sc, nPartitionOption))
  }
}
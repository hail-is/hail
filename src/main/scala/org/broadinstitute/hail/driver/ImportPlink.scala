package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.PlinkLoader
import org.kohsuke.args4j.{Option => Args4jOption}

object ImportPlink extends Command {
  def name = "importplink"

  def description = "Load PLINK binary file (.bed, .bim, .fam) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "--bfile", usage = "Plink Binary file root name", forbids = Array("--bed","--bim","--fam"))
    var bfile: String = _

    @Args4jOption(name = "--bed", usage = "Plink .bed file", forbids = Array("--bfile"), depends = Array("--bim","--fam"))
    var bed: String = _

    @Args4jOption(name = "--bim", usage = "Plink .bim file", forbids = Array("--bfile"), depends = Array("--bed","--fam"))
    var bim: String = _

    @Args4jOption(name = "--fam", usage = "Plink .fam file", forbids = Array("--bfile"), depends = Array("--bim","--bed"))
    var fam: String = _

    @Args4jOption(name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: java.lang.Integer = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val nPartitionOption = Option(options.nPartitions).map(_.toInt)

    state.hadoopConf.setBoolean("compressGS", !options.noCompress)

    if (options.bfile == null && (options.bed == null || options.bim == null || options.fam == null))
      fatal("invalid input: require either --bed/--bim/--fam arguments or --bfile argument")

    if (options.bfile != null) {
      if (options.bim != null || options.bed != null || options.fam != null)
        warn("received --bfile argument, ignoring unexpected --bed/--bim/--fam arguments")
      state.copy(vds = PlinkLoader(options.bfile + ".bed", options.bfile + ".bim", options.bfile + ".fam", state.sc, nPartitionOption))
    } else
      state.copy(vds = PlinkLoader(options.bed, options.bim, options.fam, state.sc, nPartitionOption))
  }
}
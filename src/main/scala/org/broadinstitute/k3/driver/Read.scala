package org.broadinstitute.k3.driver

import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant.VariantSampleMatrix
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.vcf
import org.kohsuke.args4j.{Option => Args4jOption}

object Read extends Command {
  def name = "read"

  def description = "Load file (.vds, .vcf or .vcf.bgz) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: String = _

    // FIXME not needed: use read vs readvcf?
    @Args4jOption(required = false, name = "-m", aliases = Array("--vsm-type"), usage = "Select VariantSampleMatrix implementation")
    var vsmtype: String = "sparky"

    @Args4jOption(required = false, name = "-p", aliases = Array("--parser"), usage = "Select parser, one of htsjdk or native")
    var parser: String = "htsjdk"

    @Args4jOption(required = false, name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(required = false, name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val input = options.input

    val parser = options.parser
    println("parser = " + parser)
    val readerBuilder = if (parser == "htsjdk")
      vcf.HtsjdkRecordReaderBuilder
    else if (parser == "native")
      vcf.RecordReaderBuilder
    else
      fatal("unknown parser `" + parser + "'")

    val newVDS =
      if (input.endsWith(".vds"))
        VariantSampleMatrix.read(state.sqlContext, input)
      else if (input.endsWith(".vcf")
        || input.endsWith(".vcf.bgz")
        || input.endsWith(".vcf.gz")) {
        LoadVCF(state.sc, input, readerBuilder, options.vsmtype, !options.noCompress,
          if (options.nPartitions != 0)
            Some(options.nPartitions)
          else
            None)
      }
      else
        fatal("unknown input file type")

    state.copy(vds = newVDS)
  }
}

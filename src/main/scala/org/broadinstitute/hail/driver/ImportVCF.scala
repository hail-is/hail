package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption, Argument}
import scala.collection.JavaConverters._

object ImportVCF extends Command {
  def name = "importvcf"

  def description = "Load file (.vcf or .vcf.bgz) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: Boolean = false

    @Args4jOption(name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(name = "-f", aliases = Array("--force"), usage = "Force load .gz file")
    var force: Boolean = false

    @Args4jOption(name = "--header-file", usage = "File to load header from")
    var headerFile: String = null

    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0

    @Args4jOption(name = "--store-gq", usage = "Store GQ instead of computing from PL")
    var storeGQ: Boolean = false

    @Argument
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    if (options.input)
      warn("-i deprecated, no longer needed")

    val inputs = options.arguments.asScala
      .iterator
      .flatMap { arg =>
        val fss = hadoopGlobAndSort(arg, state.hadoopConf)
        val files = fss.map(_.getPath.toString)
        if (files.isEmpty)
          warn(s"`$arg' refers to no files")
        files
      }.toArray

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".vcf")
        && !input.endsWith(".vcf.bgz")) {
        if (input.endsWith(".vcf.gz")) {
          if (!options.force)
            fatal(".gz cannot be loaded in parallel, use .bgz or -f override")
        } else
          fatal("unknown input file type")
      }
    }

    val headerFile = if (options.headerFile != null)
      options.headerFile
    else
      inputs.head

    state.copy(vds = LoadVCF(state.sc,
      headerFile,
      options.arguments.asScala.toArray,
      options.storeGQ,
      !options.noCompress,
      if (options.nPartitions != 0)
        Some(options.nPartitions)
      else
        None))
  }

}

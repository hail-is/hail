package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.BgenLoader
import org.kohsuke.args4j.{Option => Args4jOption, Argument}
import scala.collection.JavaConverters._

object ImportBGEN extends Command {
  def name = "importbgen"

  def description = "Load BGEN file as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0

    @Args4jOption(name = "-s", aliases = Array("--samplefile"), usage = "Sample file for BGEN files")
    var sampleFile: String = null

    @Args4jOption(name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Argument
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val nPartitions = if (options.nPartitions > 0) Some(options.nPartitions) else None

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
      if (!input.endsWith(".bgen")) {
        fatal("unknown input file type")
      }
    }

    val sampleFile = Option(options.sampleFile)

    //FIXME to be an array
    state.copy(vds = BgenLoader(inputs, sampleFile, state.sc, nPartitions, !options.noCompress))
  }
}

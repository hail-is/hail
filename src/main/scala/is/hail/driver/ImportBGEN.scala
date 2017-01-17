package is.hail.driver

import is.hail.io.bgen.BgenLoader
import is.hail.utils._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object ImportBGEN extends Command {
  def name = "importbgen"

  def description = "Load BGEN file as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0

    @Args4jOption(name = "-s", aliases = Array("--samplefile"), usage = "Sample file for BGEN files")
    var sampleFile: String = _

    @Args4jOption(name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(name = "-t", aliases = Array("--tolerance"), usage = "If abs(1 - sum dosages) > tolerance, set to None")
    var tolerance: Double = 0.02

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val nPartitions = if (options.nPartitions > 0) Some(options.nPartitions) else None

    val inputs = state.hadoopConf.globAll(options.arguments.asScala)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen")) {
        fatal("unknown input file type")
      }
    }
    val sc = state.sc

    val vds = BgenLoader.load(sc, inputs, Option(options.sampleFile), options.tolerance,
      !options.noCompress, Option(options.nPartitions))

    state.copy(vds = vds)
  }
}

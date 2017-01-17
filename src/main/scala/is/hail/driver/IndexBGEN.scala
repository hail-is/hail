package is.hail.driver

import is.hail.utils._
import is.hail.io.bgen.BgenLoader
import org.kohsuke.args4j.Argument

import scala.collection.JavaConverters._

object IndexBGEN extends Command {
  def name = "indexbgen"

  def description = "Create an index for one or more BGEN files.  `importbgen' cannot run without these indexes."

  class Options extends BaseOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {

    val inputs = state.hadoopConf.globAll(options.arguments.asScala)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen")) {
        fatal(s"unknown input file: $input")
      }
    }

    val conf = new SerializableHadoopConfiguration(state.hadoopConf)

    state.sc.parallelize(inputs).foreach { in =>
        BgenLoader.index(conf.value, in)
    }

    info(s"Number of BGEN files indexed: ${inputs.length}")

    state
  }
}

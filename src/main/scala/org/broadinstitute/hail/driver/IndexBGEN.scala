package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.BgenLoader
import org.kohsuke.args4j.{Option => Args4jOption, Argument}
import scala.collection.JavaConverters._

object IndexBGEN extends Command {
  def name = "indexbgen"

  def description = "Make an index for BGEN file. Must be done before running importbgen"

  class Options extends BaseOptions {
    @Args4jOption(name = "-s", aliases = Array("--samplefile"), usage = "Sample file for BGEN files")
    var sampleFile: String = null

    @Argument(usage = "<file>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {

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

    inputs.foreach{i => BgenLoader.createIndex(i, Option(options.sampleFile), state.sc)}

    //FIXME to be an array
    state
  }
}

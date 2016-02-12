package org.broadinstitute.hail.driver

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.variant.VariantDataset
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineException, CmdLineParser}
import scala.collection.JavaConverters._
import org.broadinstitute.hail.Utils._

case class State(sc: SparkContext,
  sqlContext: SQLContext,
  // FIXME make option
  vds: VariantDataset = null) {
  def hadoopConf = vds.sparkContext.hadoopConfiguration
}

// FIXME: HasArgs vs Command
abstract class Command {

  class BaseOptions {
    @Args4jOption(name = "-h", aliases = Array("--help"), help = true, usage = "Print usage and exit")
    var printUsage: Boolean = false
  }

  type Options <: BaseOptions

  // FIXME HACK
  def newOptions: Options

  def name: String

  def description: String

  def hidden: Boolean = false

  def supportsMultiallelic = false

  def preCheck(args: Array[String]) {
    parseArgs(args)
  }



  def parseArgs(args: Array[String]): Options = {
    val options = newOptions
    val parser = new CmdLineParser(options)

    try {
      parser.parseArgument((args: Iterable[String]).asJavaCollection)
      if (options.printUsage) {
        println("usage: " + name + " [<args>]")
        println("")
        println(description)
        println("")
        println("Arguments:")
        new CmdLineParser(newOptions).printUsage(System.out)
        sys.exit(0)
      }
    } catch {
      case e: CmdLineException =>
        println("Parse error in " + name + ": " + e.getMessage)
        sys.exit(1)
    }

    options
  }

  def run(state: State, args: Array[String]): State = {
    val options = parseArgs(args)
    if (!supportsMultiallelic
      && state.vds != null
      && !state.vds.metadata.wasSplit)
      fatal(s"`$name' does not support multiallelics.\n  Run `splitmulti' first.")

    run(state, options)
  }

  def run(state: State, options: Options): State
}

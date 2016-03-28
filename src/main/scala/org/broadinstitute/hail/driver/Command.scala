package org.broadinstitute.hail.driver

import org.apache.spark.{SparkException, SparkContext}
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.FatalException
import org.broadinstitute.hail.variant.VariantDataset
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineException, CmdLineParser}
import scala.collection.JavaConverters._
import org.broadinstitute.hail.Utils._

case class State(sc: SparkContext,
  sqlContext: SQLContext,
  // FIXME make option
  vds: VariantDataset = null) {
  def hadoopConf = sc.hadoopConfiguration
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

  def lookup(args: Array[String]): (Command, Array[String]) = (this, args)

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
        fatal(s"$name: parse error: ${e.getMessage}")
    }

    options
  }

  def runCommand(state: State, options: Options): State = {
    if (!supportsMultiallelic
      && state.vds != null
      && !state.vds.wasSplit)
      fatal(s"`$name' does not support multiallelics.\n  Run `splitmulti' first.")

    run(state, options)
  }

  def run(state: State, args: Array[String] = Array.empty[String]): State =
    runCommand(state, parseArgs(args))

  protected def run(state: State, options: Options): State
}

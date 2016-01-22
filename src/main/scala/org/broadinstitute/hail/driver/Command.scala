package org.broadinstitute.hail.driver

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.variant.VariantDataset
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineException, CmdLineParser}
import scala.collection.JavaConverters._

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
        println(e.getMessage)
        sys.exit(1)
    }

    options
  }

  def run(state: State, args: Array[String]): State = {
    val options = parseArgs(args)
    run(state, options)
  }

  def run(state: State, options: Options): State
}

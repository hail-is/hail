package org.broadinstitute.hail.driver

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.VariantDataset
import org.kohsuke.args4j.{Argument, CmdLineException, CmdLineParser, Option => Args4jOption}

import scala.collection.JavaConverters._
import scala.collection.mutable

case class State(sc: SparkContext,
  sqlContext: SQLContext,
  // FIXME make option
  vds: VariantDataset = null) {
  def hadoopConf = sc.hadoopConfiguration
}

object ToplevelCommands {
  val commands = mutable.Map.empty[String, Command]

  def commandNames: Set[String] = commands.keys.toSet

  def register(command: Command) {
    commands += command.name -> command
  }

  def lookup(args: Array[String]): (Command, Array[String]) = {
    assert(!args.isEmpty)

    val commandName = args.head
    commands.get(commandName) match {
      case Some(c) => c.lookup(args.tail)
      case None =>
        fatal(s"no such command `$commandName'")
    }
  }

  def printCommands() {
    val visibleCommands = commands.values.filterNot(_.hidden).toArray.sortBy(_.name)
    val maxLen = visibleCommands.map(_.name.length).max
    visibleCommands
      .foreach(cmd => println("  " + cmd.name + (" " * (maxLen - cmd.name.length + 2))
        + cmd.description))
  }

  register(AnnotateSamples)
  register(AnnotateVariants)
  register(AnnotateGlobal)
  register(Cache)
  register(CompareVDS)
  register(Count)
  register(DownsampleVariants)
  register(ExportPlink)
  register(ExportGenotypes)
  register(ExportSamples)
  register(ExportVariants)
  register(ExportVariantsCass)
  register(ExportVariantsSolr)
  register(ExportVCF)
  register(FilterGenotypes)
  register(FamSummary)
  register(FilterSamples)
  register(FilterVariants)
  register(GenDataset)
  register(Grep)
  register(GRM)
  register(GQByDP)
  register(GQHist)
  register(Head)
  register(ImportAnnotations)
  register(ImportVCF)
  register(ImputeSex)
  register(LinearRegressionCommand)
  register(MendelErrorsCommand)
  register(SplitMulti)
  register(PCA)
  register(Persist)
  register(Read)
  register(ReadKudu)
  register(RenameSamples)
  register(Repartition)
  register(SampleQC)
  register(PrintSchema)
  register(ShowGlobalAnnotations)
  register(VariantQC)
  register(VEP)
  register(Write)
  register(WriteKudu)

  // example commands
  register(example.CaseControlCount)
}

abstract class SuperCommand extends Command {

  class Options extends BaseOptions {
    @Argument(required = true, usage = "<subcommand> <arguments...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  val subcommands = mutable.Map.empty[String, Command]

  def subcommandNames: Set[String] = subcommands.keys.toSet

  def register(subcommand: Command) {
    val split = subcommand.name.split(" ")
    assert(name == split.init.mkString(" "))
    subcommands += split.last -> subcommand
  }

  def supportsMultiallelic = true

  def requiresVDS = false

  override def lookup(args: Array[String]): (Command, Array[String]) = {
    val subArgs = args.dropWhile(_ == "-h")

    if (subArgs.isEmpty)
      return (this, args)

    val subcommandName = subArgs.head
    subcommands.get(subcommandName) match {
      case Some(sc) => sc.lookup(args.tail)
      case None =>
        fatal(s"$name: no such sub-command `$subcommandName'")
    }
  }

  override def printUsage() {
    super.printUsage()

    println("")
    println("Sub-commands:")
    val visibleSubcommands = subcommands.values.filterNot(_.hidden).toArray.sortBy(_.name)
    val maxLen = visibleSubcommands.map(_.name.length).max
    visibleSubcommands
      .foreach(sc => println("  " + sc.name + (" " * (maxLen - sc.name.length + 2))
        + sc.description))
  }

  override def parseArgs(args: Array[String]): Options = {
    val options = newOptions
    if (args(0) == "-h")
      options.printUsage = true

    val subArgs = args.dropWhile(_ == "-h")
    options.arguments = new java.util.ArrayList[String](subArgs.toList.asJava)

    if (options.printUsage) {
      printUsage()
      sys.exit(0)
    }

    options
  }

  override def runCommand(state: State, options: Options): State = {
    run(state, options)
  }

  def run(state: State, options: Options): State = {
    val args = options.arguments.asScala.toArray

    if (args.isEmpty)
      fatal(s"no sub-command given.  See help (-h) output for list of sub-commands.")

    val (sc, scArgs) = lookup(args)
    sc.run(state, scArgs)
  }
}

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

  def supportsMultiallelic: Boolean

  def requiresVDS: Boolean

  def lookup(args: Array[String]): (Command, Array[String]) = (this, args)

  def printUsage() {
    println("usage: " + name + " [<args>]")
    println("")
    println(description)
    println("")
    println("Arguments:")
    new CmdLineParser(newOptions).printUsage(System.out)
  }

  def parseArgs(args: Array[String]): Options = {
    val options = newOptions
    val parser = new CmdLineParser(options)

    try {
      parser.parseArgument((args: Iterable[String]).asJavaCollection)
      if (options.printUsage) {
        printUsage()
        sys.exit(0)
      }
    } catch {
      case e: CmdLineException =>
        fatal(s"parse error: ${e.getMessage}")
    }

    options
  }

  def runCommand(state: State, options: Options): State = {
    if (requiresVDS && state.vds == null)
      fatal("this module requires a VDS.\n  Provide a VDS through a `read' or `import' command first.")
    else if (!supportsMultiallelic && !state.vds.wasSplit)
      fatal("this module does not support multiallelic variants.\n  Please run `splitmulti' first.")
    else
      run(state, options)
  }

  def run(state: State, args: Array[String] = Array.empty): State =
    runCommand(state, parseArgs(args))

  protected def run(state: State, options: Options): State
}

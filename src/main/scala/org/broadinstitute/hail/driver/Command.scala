package org.broadinstitute.hail.driver

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.VariantDataset
import org.kohsuke.args4j.{Argument, CmdLineException, CmdLineParser, Option => Args4jOption}

import scala.collection.JavaConverters._
import scala.collection.mutable

case class State(sc: SparkContext,
  sqlContext: SQLContext,
  // FIXME make option
  vds: VariantDataset = null,
  env: Map[String, VariantDataset] = Map.empty,
  ktEnv: Map[String, KeyTable] = Map.empty) {
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

  register(AggregateByKey)
  register(AggregateIntervals)
  register(AnnotateSamples)
  register(AnnotateVariants)
  register(AnnotateGlobal)
  register(BaldingNicholsModelCommand)
  register(BGZipBlocks)
  register(Cache)
  register(Clear)
  register(CommandMetadata)
  register(CompareVDS)
  register(Concordance)
  register(Count)
  register(CountBytes)
  register(Deduplicate)
  register(DownsampleVariants)
  register(ExportPlink)
  register(ExportGEN)
  register(ExportGenotypes)
  register(ExportSamples)
  register(ExportVariants)
  register(ExportVariantsCass)
  register(ExportVariantsSolr)
  register(ExportVCF)
  register(FilterAlleles)
  register(FilterGenotypes)
  register(Filtermulti)
  register(FilterSamples)
  register(FilterVariants)
  register(GenDataset)
  register(Get)
  register(Grep)
  register(GRM)
  register(GQByDP)
  register(GQHist)
  register(HardCalls)
  register(IBDCommand)
  register(ImportAnnotations)
  register(ImportBGEN)
  register(ImportGEN)
  register(ImportPlink)
  register(ImportVCF)
  register(ImputeSex)
  register(IndexBGEN)
  register(Join)
  register(LinearRegressionCommand)
  register(LogisticRegressionCommand)
  register(MendelErrorsCommand)
  register(SparkInfo)
  register(SplitMulti)
  register(PCA)
  register(Persist)
  register(Put)
  register(Read)
  register(ReadKudu)
  register(RenameSamples)
  register(Coalesce)
  register(SampleQC)
  register(SeqrServerCommand)
  register(TDTCommand)
  register(Typecheck)
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
        fatal(s"parse error: ${ e.getMessage }")
    }

    options
  }

  def runCommand(state: State, options: Options): State = {
    if (requiresVDS && state.vds == null)
      fatal("this module requires a VDS.\n  Provide a VDS through a `read' or `import' command first.")
    else if (!supportsMultiallelic && !state.vds.wasSplit)
      fatal("this module does not support multiallelic variants.\n  Please run `splitmulti' first.")
    else {
      if (requiresVDS)
        log.info(s"sparkinfo: $name, ${ state.vds.nPartitions } partitions, ${ state.vds.rdd.getStorageLevel.toReadableString() }")
      run(state, options)
    }
  }

  def run(state: State, args: Array[String] = Array.empty): State =
    runCommand(state, parseArgs(args))

  def run(state: State, args: String*): State = run(state, args.toArray)

  protected def run(state: State, options: Options): State
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.bgen.BgenLoader
import org.broadinstitute.hail.variant._
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

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val nPartitions = if (options.nPartitions > 0) Some(options.nPartitions) else None

    val inputs = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen")) {
        fatal("unknown input file type")
      }
    }
    val sc = state.sc

    val samples = Option(options.sampleFile) match {
      case Some(file) => BgenLoader.readSampleFile(sc.hadoopConfiguration, file)
      case _ => BgenLoader.readSamples(sc.hadoopConfiguration, inputs.head)
    }

    val nSamples = samples.length

    sc.hadoopConfiguration.setBoolean("compressGS", !options.noCompress)

    val results = inputs.map(f => BgenLoader.load(sc, f, Option(options.nPartitions)))

    val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
    if (unequalSamples.length > 0)
      fatal(
        s"""The following BGEN files did not contain the expected number of samples $nSamples:
            |  ${unequalSamples.map(x => s"""(${x._2} ${x._1}""").mkString("\n  ")}""".stripMargin)

    val noVariants = results.filter(_.nVariants == 0).map(_.file)
    if (noVariants.length > 0)
      fatal(
        s"""The following BGEN files did not contain at least 1 variant:
            |  ${noVariants.mkString("\n  ")})""".stripMargin)

    val nVariants = results.map(_.nVariants).sum

    info(s"Number of BGEN files parsed: ${results.length}")
    info(s"Number of variants in all BGEN files: $nVariants")
    info(s"Number of samples in BGEN files: $nSamples")

    val signature = TStruct("rsid" -> TString, "varid" -> TString)

    val rdd = sc.union(results.map(_.rdd))
    val vds = VariantSampleMatrix(VariantMetadata(samples), rdd).copy(vaSignature = signature, wasSplit = true)

    state.copy(vds = vds)
  }
}

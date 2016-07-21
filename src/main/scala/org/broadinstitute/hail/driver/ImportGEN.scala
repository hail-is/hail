package org.broadinstitute.hail.driver

import org.apache.spark.rdd.OrderedRDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.bgen.BgenLoader
import org.broadinstitute.hail.io.gen.GenLoader
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object ImportGEN extends Command {
  def name = "importgen"

  def description = "Load GEN file as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0

    @Args4jOption(name = "-s", required = true, aliases = Array("--samplefile"), usage = "Sample file for GEN files")
    var sampleFile: String = null

    @Args4jOption(name = "-c", aliases = Array("--chromosome"), usage = "Chromosome if not listed in GEN file")
    var chromosome: String = null

    @Args4jOption(name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(name = "-t", aliases = Array("--tolerance"), usage = "If sum dosages < (1 - tolerance), set to None")
    var tolerance: Double = 0.02

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val nPartitions = if (options.nPartitions > 0) Some(options.nPartitions) else None

    val inputs = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".gen")) {
        fatal("unknown input file type")
      }
    }

    val sc = state.sc

    val samples = BgenLoader.readSampleFile(sc.hadoopConfiguration, options.sampleFile)

    val nSamples = samples.length

    //FIXME: can't specify multiple chromosomes
    val results = inputs.map(f => GenLoader(f, options.sampleFile, sc, Option(options.nPartitions),
      options.tolerance, !options.noCompress, Option(options.chromosome)))

    val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
    if (unequalSamples.length > 0)
      fatal(
        s"""The following GEN files did not contain the expected number of samples $nSamples:
            |  ${ unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ") }""".stripMargin)

    val noVariants = results.filter(_.nVariants == 0).map(_.file)
    if (noVariants.length > 0)
      fatal(
        s"""The following GEN files did not contain at least 1 variant:
            |  ${ noVariants.mkString("\n  ") })""".stripMargin)

    val nVariants = results.map(_.nVariants).sum

    info(s"Number of GEN files parsed: ${ results.length }")
    info(s"Number of variants in all GEN files: $nVariants")
    info(s"Number of samples in GEN files: $nSamples")

    val signature = TStruct("rsid" -> TString, "varid" -> TString)

    val vds = VariantSampleMatrix(VariantMetadata(samples).copy(isDosage = true),
      sc.union(results.map(_.rdd)).toOrderedRDD(_.locus))
      .copy(vaSignature = signature, wasSplit = true)

    state.copy(vds = vds)
  }
}


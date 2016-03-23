package org.broadinstitute.hail.driver

import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StructType}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.variant.{GenotypeStream, GenotypeStreamBuilder, Variant, VariantDataset, VariantMetadata}
import org.kohsuke.args4j.{Option => Args4jOption}

object PreprocessAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--import"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Path for writing annotation VDS")
    var output: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = "NA"

    @Args4jOption(required = false, name = "--vcolumns",
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)" +
        " (default: 'Chromosome,Position,Ref,Alt'")
    var vCols: String = "Chromosome, Position, Ref, Alt"
  }

  def newOptions = new Options

  def name = "preprocessannotations"

  def description = "Convert a tsv file containing variant annotations into a sample-free vds"

  def run(state: State, options: Options): State = {
    val out = options.output
    if (!out.endsWith(".vds"))
      fatal("Output path must end in '.vds'")

    val cond = options.condition

    val conf = state.sc.hadoopConfiguration
    val serializer = SparkEnv.get.serializer.newInstance()

    val vds = hadoopStripCodec(options.condition, conf) match {
      case tsv if tsv.endsWith(".tsv") =>
        val (rdd, signature) = VariantTSVAnnotator(state.sc, cond, AnnotateVariants.parseColumns(options.vCols),
          AnnotateVariants.parseTypeMap(Option(options.types).getOrElse("")), AnnotateVariants.parseMissing(options.missingIdentifiers))
        new VariantDataset(
          VariantMetadata(IndexedSeq.empty[(String, String)], Array.empty[String], Annotation.emptyIndexedSeq(0),
            expr.TEmpty, signature, wasSplit = true),
          Array.empty[Int],
          rdd.map { case (v, va) => (v, va, new GenotypeStreamBuilder(v, true).result()) })
      case _ =>
        fatal(
          """This module requires an input file in the following format:
            |  .tsv (tab separated values with chr, pos, ref, and alt columns, or chr:pos:ref:alt column)""".stripMargin)
    }

    hadoopDelete(out, state.hadoopConf, recursive = true)

    vds.write(state.sqlContext, out)

    state
  }
}

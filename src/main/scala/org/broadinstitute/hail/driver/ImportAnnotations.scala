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

object ImportAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Annotation file path")
    var input: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = ""

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing")
    var missingIdentifiers: String = "NA"

    @Args4jOption(required = false, name = "--vcolumns",
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)")
    var vCols: String = "Chromosome, Position, Ref, Alt"
  }

  def newOptions = new Options

  def name = "importannotations"

  def description = "Import a tsv file containing variants / annotations into a sample-free vds"

  def run(state: State, options: Options): State = {
    val cond = options.input

    val conf = state.sc.hadoopConfiguration
    val serializer = SparkEnv.get.serializer.newInstance()

    val vds = hadoopStripCodec(options.input, conf) match {
      case tsv if tsv.endsWith(".tsv") =>
        val (rdd, signature) = VariantTSVAnnotator(state.sc, cond, AnnotateVariantsTSV.parseColumns(options.vCols),
          AnnotateVariantsTSV.parseTypeMap(options.types), options.missingIdentifiers)
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

    state.copy(vds = vds)
  }
}

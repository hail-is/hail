package org.broadinstitute.hail.driver

import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StructType}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.variant.{GenotypeStream, Variant}
import org.kohsuke.args4j.{Option => Args4jOption}

object ConvertAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--import"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Path for writing write serialized file")
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

  def name = "convertannotations"

  def description = "Convert a tsv or vcf file containing variant annotations into the fast hail format"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val out = options.output
    if (!out.endsWith(".faf"))
      fatal("Output path must end in 'faf'")

    val cond = options.condition

    val conf = state.sc.hadoopConfiguration
    val serializer = SparkEnv.get.serializer.newInstance()

    val (rdd, signature) = hadoopStripCodec(options.condition, conf) match {
      case tsv if tsv.endsWith(".tsv") =>
        TSVAnnotator(state.sc, cond, AnnotateVariants.parseColumns(options.vCols),
          AnnotateVariants.parseTypeMap(options.types), AnnotateVariants.parseMissing(options.missingIdentifiers))
      case vcf if vcf.endsWith(".vcf") =>
        VCFAnnotator(state.sc, cond)
      case _ =>
        fatal(
          """This module requires an input file ending in one of the following:
            |  .tsv (tab separated values with chr, pos, ref, alt)
            |  .vcf (vcf, only the info field / filters / qual are parsed here)""".stripMargin)
    }


    hadoopMkdir(out, conf)
    writeDataFile(out + "/signature.ser", conf) {
      dos => {
        val serializer = SparkEnv.get.serializer.newInstance()
        serializer.serializeStream(dos).writeObject(signature)
      }
    }

    val schema = StructType(Array(
      StructField("variant", Variant.schema(), false),
      StructField("annotation", signature.getSchema, true)
    ))

    state.sqlContext.createDataFrame(rdd.map { case (v, a) => Row.fromSeq(Array((Variant.toRow(v), a))) }, schema)
      .write.parquet(out + "/rdd.parquet")
    // .saveAsParquetFile(dirname + "/rdd.parquet")

  state
}

import org.apache.spark.rdd.HadoopRDD

}

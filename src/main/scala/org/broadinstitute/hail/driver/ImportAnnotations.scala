package org.broadinstitute.hail.driver

import org.apache.spark.rdd.OrderedRDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.{TextTableOptions, TextTableReader}
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object ImportAnnotations extends SuperCommand {
  def name = "importannotations"

  def description = "Import variants and annotations as a sites-only VDS"

  register(ImportAnnotationsTable)
}

object ImportAnnotationsTable extends Command with JoinAnnotator {

  class Options extends BaseOptions with TextTableOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

    @Args4jOption(required = true, name = "-e", aliases = Array("--variant-expr"),
      usage = "Specify an expression to construct a variant from the fields of the text table")
    var vExpr: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions select specific columns / groups")
    var code: String = _
  }

  def newOptions = new Options

  def name = "importannotations table"

  def description = "Import variants and annotations from a delimited text file as a sites-only VDS"

  def requiresVDS = false

  def supportsMultiallelic = true

  def run(state: State, options: Options): State = {

    val files = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, rdd) = TextTableReader.read(state.sc, files, options.config)

    val (finalType, fn): (Type, (Annotation, Option[Annotation]) => Annotation) = Option(options.code).map { code =>
      val ec = EvalContext(Map(
        "va" -> (0, TStruct.empty),
        "table" -> (1, struct)))
      buildInserter(code, TStruct.empty, ec, Annotation.VARIANT_HEAD)
    }.getOrElse((struct, (_: Annotation, anno: Option[Annotation]) => anno.orNull))

    val ec = EvalContext(struct.fields.map(f => (f.name, f.`type`)): _*)
    val variantFn = Parser.parse[Variant](options.vExpr, ec, TVariant)

    val keyedRDD = rdd.flatMap {
      _.map { a =>
        ec.setAll(a.asInstanceOf[Row].toSeq: _*)
        variantFn().map(v => (v, (fn(null, Some(a)), Iterable.empty[Genotype])))
      }.value
    }.toOrderedRDD(_.locus)

    val vds: VariantDataset = VariantSampleMatrix(VariantMetadata(Array.empty[String], IndexedSeq.empty[Annotation], Annotation.empty,
      TStruct.empty, finalType, TStruct.empty), keyedRDD)

    state.copy(vds = vds)
  }

}

package org.broadinstitute.hail.driver

import org.apache.spark.sql.Row
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Locus, Variant}
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object AnnotateVariantsTable extends Command with JoinAnnotator {

  class Options extends BaseOptions with TextTableOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va' (this argument or --code required)")
    var root: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions to join with the table (this argument or --root required)")
    var code: String = _

    @Args4jOption(required = true, name = "-e", aliases = Array("--variant-expr"),
      usage = "Specify an expression to construct a variant from the fields of the text table")
    var vExpr: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants table"

  def description = "Annotate variants with delimited text file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {

    val files = state.hadoopConf.globAll(options.arguments.asScala)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val vds = state.vds

    val (expr, code) = (Option(options.code), Option(options.root)) match {
      case (Some(c), None) => (true, c)
      case (None, Some(r)) => (false, r)
      case _ => fatal("this module requires one of `--root' or `--code', but not both")
    }

    val (struct, rdd) = TextTableReader.read(state.sc)(files, options.config, vds.nPartitions)

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (expr) {
      val ec = EvalContext(Map(
        "va" -> (0, vds.vaSignature),
        "table" -> (1, struct)))
      buildInserter(code, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
    } else vds.insertVA(struct, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))

    val variantQuery = struct.parseInStructScope[Variant](options.vExpr)

    val keyedRDD = rdd.flatMap {
      _.map { a =>
        variantQuery(a).map(v => (v, a))
      }.value
    }.toOrderedRDD(vds.rdd.orderedPartitioner)

    state.copy(vds = vds
      .annotateVariants(keyedRDD, finalType, inserter))
  }
}
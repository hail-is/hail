package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Locus
import org.broadinstitute.hail.variant.LocusImplicits.orderedKey
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object AnnotateVariantsLoci extends Command with JoinAnnotator {

  class Options extends BaseOptions with TextTableOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va' (this argument or --code required)")
    var root: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions to join with the table (this argument or --root required)")
    var code: String = _

    @Args4jOption(required = true, name = "-e", aliases = Array("--locus-expr"),
      usage = "Specify an expression to construct a locus")
    var locusExpr: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants loci"

  def description = "Annotate variants by locus (chromosome, position) with delimited text file"

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

    val locusQuery = struct.parseInStructScope[Locus](options.locusExpr, TLocus)

    val lociRDD = rdd.flatMap {
      _.map { a =>
        locusQuery(a).map(l => (l, a))
      }.value
    }.toOrderedRDD(vds.rdd.orderedPartitioner.mapMonotonic)

    state.copy(vds = vds
      .withGenotypeStream()
      .annotateLoci(lociRDD, finalType, inserter))
  }
}
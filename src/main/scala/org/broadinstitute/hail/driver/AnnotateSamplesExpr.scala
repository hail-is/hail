package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, Inserter}
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr.Type
import org.broadinstitute.hail.io.annotators.SampleTSVAnnotator
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateSamplesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation condition")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples expr"

  def description = "Annotate samples programatically"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val symTab = Map(
      "s" ->(0, expr.TSample),
      "sa" ->(1, vds.saSignature))
    val a = new mutable.ArrayBuffer[Any](2)
    for (_ <- symTab)
      a += null

    val parsed = expr.Parser.parseAnnotationArgs(symTab, a, cond)
    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "sa")
        fatal(s"Path must start with `sa.', got `${ids.mkString(".")}'")
      val sig = t match {
        case tws: Type => tws
        case _ => fatal(s"got an invalid type `$t' from the result of `${ids.mkString(".")}'")
      }
      (ids.tail, sig)
    }
    val computations = parsed.map(_._3)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>

      val (s, i) = v.insertSA(signature, ids)
      inserterBuilder += i
      v.copy(saSignature = s)
    }
    val inserters = inserterBuilder.result()

    val newAnnotations = vdsAddedSigs.sampleAnnotations.zipWithIndex.map { case (sa, i) =>
      a(0) = Sample(vds.sampleIds(i))
      a(1) = sa

      val queries = computations.map(_ ())
      var newSA = sa
      queries.indices.foreach { i =>
        newSA = inserters(i)(newSA,
          Option(queries(i)))
      }
      newSA
    }
    val annotated = vdsAddedSigs.copy(sampleAnnotations = newAnnotations)
    state.copy(vds = annotated)
  }
}

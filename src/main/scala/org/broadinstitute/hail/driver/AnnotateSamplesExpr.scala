package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, Inserter}
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators.SampleTSVAnnotator
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object AnnotateSamplesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation condition")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples/expr"

  def description = "Annotate samples in current dataset"

  override def hidden = true

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition
    val symTab = Map(
      "s" ->(0, TSample),
      "sa" ->(1, vds.saSignature),
      "gs" ->(2, TGenotypeStream))
    val aggregationTable = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)
    )

    val ec = EvalContext(symTab, aggregationTable)
    val parsed = expr.Parser.parseAnnotationArgs(ec, cond)

    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "sa")
        fatal(s"expect 'sa[.identifier]+', got ${ids.mkString(".")}")
      (ids.tail, t)
    }
    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertSA(signature, ids)
      inserterBuilder += i
      v.copy(saSignature = s)
    }

    val computations = parsed.map(_._3)
    val inserters = inserterBuilder.result()
    
    val a = ec.a
    val aggregatorA = ec.aggregatorA

    val aggregatorArray = Aggregators.buildSampleAggregations(vds, ec)

    val newAnnotations = vdsAddedSigs.sampleAnnotations.zipWithIndex.map { case (sa, i) =>
      a(0) = Sample(vds.sampleIds(i))
      a(1) = sa
      a(2) = 0 //FIXME placeholder?

      aggregatorArray.foreach {arr =>
        arr(i).iterator.zipWithIndex
          .foreach { case (value, j) =>
            aggregatorA(5 + j) = value }
      }

      val queries = computations.map(_ ())
      queries.indices.foreach { i =>
        a(1) = inserters(i).apply(
          a(1),
          Option(queries(i)))
      }
      a(1): Annotation
    }
    val annotated = vdsAddedSigs.copy(sampleAnnotations = newAnnotations)
    state.copy(vds = annotated)
  }
}

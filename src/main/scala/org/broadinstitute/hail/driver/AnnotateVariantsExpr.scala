package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object AnnotateVariantsExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants/expr"

  override def hidden = true

  def description = "Annotate variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "gs" ->(2, TGenotypeStream))
    val aggregationTable = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)
    )

    val ec = EvalContext(symTab, ("gs", EvalContext(aggregationTable)))
    val parsed = expr.Parser.parseAnnotationArgs(ec, cond)


    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "va")
        fatal(s"expect 'va[.identifier]+', got ${ids.mkString(".")}")
      (ids.tail, t)
    }

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val computations = parsed.map(_._3)

    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertVA(signature, ids)
      inserterBuilder += i
      v.copy(vaSignature = s)
    }

    val inserters = inserterBuilder.result()

    val a = ec.a
    val aggregatorA = ec.children("gs").a

    val aggregateOption = Aggregators.buildVariantaggregations(vds, ec, "gs")

    val annotated = vdsAddedSigs.mapAnnotations { case (v, va, gs) =>
      a(0) = v
      a(1) = va
      a(2) = gs

      aggregateOption.foreach(f => f(v, va, gs))

      computations.indices.foreach { i =>
        a(1) = inserters(i).apply(a(1), Option(computations(i)()))
      }
      a(1): Annotation
    }
    state.copy(vds = annotated)
  }

}

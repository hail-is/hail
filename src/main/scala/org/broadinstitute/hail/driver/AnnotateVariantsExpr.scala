package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateVariantsExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants expr"

  def description = "Annotate variants programatically"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition


    val aggregationEC = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)))
    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "gs" ->(-1, TAggregable(aggregationEC)))


    val ec = EvalContext(symTab)
    val parsed = Parser.parseAnnotationArgs(cond, ec)

    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "va")
        fatal(s"Path must start with `va.', got `${ids.mkString(".")}'")
      val sig = t match {
        case tws: Type => tws
        case _ => fatal(s"got an invalid type `$t' from the result of `${ids.mkString(".")}'")
      }
      (ids.tail, sig)
    }
    val computations = parsed.map(_._3)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertVA(signature, ids)
      inserterBuilder += i
      v.copy(vaSignature = s)
    }
    val inserters = inserterBuilder.result()

    val a = ec.a

    val aggregateOption = Aggregators.buildVariantaggregations(vds, aggregationEC)

    val annotated = vdsAddedSigs.mapAnnotations { case (v, va, gs) =>
      a(0) = v
      a(1) = va

      aggregateOption.foreach(f => f(v, va, gs))

      var newVA = va
      computations.indices.foreach { i =>
        newVA = inserters(i)(newVA, computations(i)())
      }
      newVA
    }
    state.copy(vds = annotated)
  }

}

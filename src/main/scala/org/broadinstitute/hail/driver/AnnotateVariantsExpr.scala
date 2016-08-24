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

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition


    val aggregationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "global" -> (4, vds.globalSignature)))
    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "global" -> (2, vds.globalSignature),
      "gs" -> (-1, BaseAggregable(aggregationEC, TGenotype)))


    val ec = EvalContext(symTab)
    ec.set(2, vds.globalAnnotation)
    aggregationEC.set(4, vds.globalAnnotation)

    val (parseTypes, fns) = Parser.parseAnnotationArgs(cond, ec, Annotation.VARIANT_HEAD)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = parseTypes.foldLeft(vds.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vds, aggregationEC)

    val annotated = vds.mapAnnotations { case (v, va, gs) =>
      ec.setAll(v, va)

      aggregateOption.foreach(f => f(v, va, gs))
      fns.zip(inserters)
        .foldLeft(va) { case (va, (fn, inserter)) =>
          inserter(va, fn())
        }
    }.copy(vaSignature = finalType)
    state.copy(vds = annotated)
  }
}

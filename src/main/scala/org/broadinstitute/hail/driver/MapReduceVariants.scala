package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.variant.{Genotype, Sample, Variant}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object MapReduceVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "mapreduce"

  def description = "Annotate global table"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val aggECV2 = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)))
    val aggECS2 = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)))

    val aggECV = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "gs" ->(2, TAggregable(aggECV2))))
    val aggECS = EvalContext(Map(
      "s" ->(0, TSample),
      "sa" ->(1, vds.saSignature)
//      "gs" ->(2, TGenotypeStream)
    ))
    val symTab = Map(
      "a" ->(0, vds.taSignature),
      "vas" ->(1, TAggregable(aggECV)),
      "sas" ->(2, TAggregable(aggECS)))


//    val ec = EvalContext(symTab, ("gs", EvalContext(aggregationTable)))
//
//    val aggregationEC2 = EvalContext(aggregationTable2)
//    val aggregationEC = EvalContext(aggregationTable, aggregationEC2)
    val ec = EvalContext(symTab)
    val parsed = expr.Parser.parseAnnotationArgs(ec, cond)


    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "a")
        fatal(s"expect 'a[.identifier]+', got ${ids.mkString(".")}")
      (ids.tail, t)
    }

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val computations = parsed.map(_._3)

    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertTA(signature, ids)
      inserterBuilder += i
      v.copy(taSignature = s)
    }

    val inserters = inserterBuilder.result()

    val a = ec.a
    val sampleA = aggECV.a
    val variantA = aggECS.a

    val sampleAggregateOption = Aggregators.buildSampleAggregations(vds, aggECS)
    val variantAggregateOption = Aggregators.buildVariantaggregations(vds, aggECV)

    val toplevelAggregations = ec.aggregationFunctions.toArray

    val seqOps = toplevelAggregations.map(_._2)
    val combOps = toplevelAggregations.map(_._3)


    val aaa = vds.rdd.treeAggregate(toplevelAggregations.map(_._1()))({
      case (arr, (v, va, gs)) =>
        variantA(0) = v
        variantA(1) = va
        variantA(2) = gs

        variantAggregateOption.foreach(f => f(v, va, gs))

        arr.iterator.zip(seqOps.iterator)
          .map { case (value, so) =>
            so(value)
          }
          .toArray
    }, {
      case (arr1, arr2) =>
        arr1.iterator
          .zip(arr2.iterator)
          .zip(combOps.iterator)
          .map { case ((arr1i, arr2i), co) =>
            co(arr1i, arr2i)
          }
          .toArray
    })

    a(0) = vds.globalAnnotation
    a(1) = 5 //FIXME placeholder

    aaa.zipWithIndex.foreach {
      case (value, i) =>
        a(2 + i) = value
    }


    val ga = inserters.zipWithIndex
      .foldLeft(vds.globalAnnotation) {
        case (anno, (ins, index)) =>
          ins(anno, Option(computations(index)()))
      }


    state.copy(vds = vdsAddedSigs.copy(tAnnotation = ga))
  }

}


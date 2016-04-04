package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.variant.{Genotype, Sample, Variant}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object MapReduce extends Command {

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

    //    val aggECV2 = EvalContext(Map(
    //      "v" ->(0, TVariant),
    //      "va" ->(1, vds.vaSignature),
    //      "s" ->(2, TSample),
    //      "sa" ->(3, vds.saSignature),
    //      "g" ->(4, TGenotype)))
    //    val aggECS2 = EvalContext(Map(
    //      "v" ->(0, TVariant),
    //      "va" ->(1, vds.vaSignature),
    //      "s" ->(2, TSample),
    //      "sa" ->(3, vds.saSignature),
    //      "g" ->(4, TGenotype)))

    val aggECV = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature)))
    //      "gs" ->(2, TAggregable(aggECV2))))
    val aggECS = EvalContext(Map(
      "s" ->(0, TSample),
      "sa" ->(1, vds.saSignature)))
    //      "gs" ->(2, TAggregable(aggECS2))
    val symTab = Map(
      "a" ->(0, vds.taSignature),
      "variants" ->(-1, TAggregable(aggECV)),
      "samples" ->(-1, TAggregable(aggECS)))


    val ec = EvalContext(symTab)
    val parsed = expr.Parser.parseAnnotationArgs(ec, cond)

//    println(
//      s"""parsed the thing.  Got:
//          |  # sample folds: ${aggECS.aggregationFunctions.length}
//          |  # variant folds: ${aggECV.aggregationFunctions.length}
//      """.stripMargin)

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

    a(0) = vds.globalAnnotation


    /**
      * PLAN
      *
      * 1.  CHECK TO SEE IF VARIANT REDUCTIONS IS NONZERO
      * 2.  CHECK TO SEE IF SAMPLE GS AGGREGATIONS IS NONZERO
      * ----> if either of the above is nonzero, need to do an aggregateByVariants
      *
      * 3.  Do aggregateByVariants (if needed), add results to T and SA
      * 4.  Do sample aggregations (if needed), add results to T
      *
      *
      */

    val doVariantAgg = aggECV.aggregationFunctions.nonEmpty

    val vAgg = aggECV.aggregationFunctions

    if (vAgg.nonEmpty) {
//      println("doing variant agg")
      val vArray = aggECV.a
      val zVals = aggECV.aggregationFunctions.map(_._1.apply()).toArray
      val seqOps = aggECV.aggregationFunctions.map(_._2).toArray
      val combOps = aggECV.aggregationFunctions.map(_._3).toArray
      val indices = aggECV.aggregationFunctions.map(_._4).toArray
      val sampleInfoBc = vds.sparkContext.broadcast(
        vds.localSamples.map(vds.sampleAnnotations)
          .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))
      val result = vds.variantsAndAnnotations
        .treeAggregate(zVals)({ case (arr, (v, va)) =>
          vArray(0) = v
          vArray(1) = va

          for (i <- arr.indices) {
            val seqOp = seqOps(i)
            arr(i) = seqOp(arr(i))
          }
          arr
        }, { case (arr1, arr2) =>
          for (i <- arr1.indices) {
            val combOp = combOps(i)
            arr1(i) = combOp(arr1(i), arr2(i))
          }
          arr1
        })
//
      result.iterator
        .zip(indices.iterator)
        .foreach { case (res, index) =>
          vArray(index) = res
        }
    }

    val sAgg = aggECS.aggregationFunctions

    if (sAgg.nonEmpty) {
//      println("doing sample agg")
      val sArray = aggECS.a
      val zVals = aggECS.aggregationFunctions.map(_._1.apply()).toArray
      val seqOps = aggECS.aggregationFunctions.map(_._2).toArray
      val combOps = aggECS.aggregationFunctions.map(_._3).toArray
      val indices = aggECS.aggregationFunctions.map(_._4).toArray
      val sampleInfoBc = vds.sparkContext.broadcast(
        vds.localSamples.map(vds.sampleAnnotations)
          .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))
      val result = vds.localSamples.map(i => (vds.sampleIds(i), vds.sampleAnnotations(i)))
          .aggregate(zVals)({ case (arr, (s, sa)) =>
          sArray(0) = s
          sArray(1) = sa

          for (i <- arr.indices) {
            val seqOp = seqOps(i)
            arr(i) = seqOp(arr(i))
          }
          arr
        }, { case (arr1, arr2) =>
          for (i <- arr1.indices) {
            val combOp = combOps(i)
            arr1(i) = combOp(arr1(i), arr2(i))
          }
          arr1
        })

      result.iterator
        .zip(indices.iterator)
        .foreach { case (res, index) =>
          sArray(index) = res
        }
    }

    val ga = inserters
      .zip(parsed.map(_._3()))
      .foldLeft(vds.globalAnnotation){ case (a, (ins, res)) =>
      ins(a, Option(res))
      }

    state.copy(
      vds = vdsAddedSigs.copy(tAnnotation = ga)
    )
  }
}


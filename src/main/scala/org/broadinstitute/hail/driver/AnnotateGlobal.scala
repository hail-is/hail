package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateGlobal extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotateglobal"

  def description = "Annotate global table"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val aggECV = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature)))
    val aggECS = EvalContext(Map(
      "s" ->(0, TSample),
      "sa" ->(1, vds.saSignature)))
    val symTab = Map(
      "global" ->(0, vds.globalSignature),
      "variants" ->(-1, TAggregable(aggECV)),
      "samples" ->(-1, TAggregable(aggECS)))


    val ec = EvalContext(symTab)
    val parsed = expr.Parser.parseAnnotationArgs(cond, ec)

    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "global")
        fatal(s"Path must start with `global', got `${ids.mkString(".")}'")
      (ids.tail, t)
    }

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val computations = parsed.map(_._3)

    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertGlobal(signature, ids)
      inserterBuilder += i
      v.copy(globalSignature = s)
    }

    val inserters = inserterBuilder.result()

    val a = ec.a
    val sampleA = aggECV.a
    val variantA = aggECS.a

    val vAgg = aggECV.aggregationFunctions.toArray

    if (vAgg.nonEmpty) {
      val vArray = aggECV.a
      val zVals = vAgg.map(_._1.apply())
      val seqOps = vAgg.map(_._2)
      val combOps = vAgg.map(_._3)
      val indices = vAgg.map(_._4)

      val result = vds.variantsAndAnnotations
        .treeAggregate(zVals)({ case (arr, (v, va)) =>
          aggECV.setContext(v, va)
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

    val sAgg = aggECS.aggregationFunctions.toArray

    if (sAgg.nonEmpty) {
      val sArray = aggECS.a
      val zVals = sAgg.map(_._1.apply())
      val seqOps = sAgg.map(_._2)
      val combOps = sAgg.map(_._3)
      val indices = sAgg.map(_._4)
      val result = vds.sampleIdsAndAnnotations
        .aggregate(zVals)({ case (arr, (s, sa)) =>
          aggECS.setContext(s, sa)

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

    a(0) = vds.globalAnnotation

    val ga = inserters
      .zip(parsed.map(_._3()))
      .foldLeft(vds.globalAnnotation) { case (a, (ins, res)) =>
        ins(a, res)
      }

    state.copy(
      vds = vdsAddedSigs.copy(globalAnnotation = ga)
    )
  }
}


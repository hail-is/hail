package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.MultiArray2
import org.broadinstitute.hail.variant._

object Aggregators {

  def buildVariantAggregations(vds: VariantDataset, ec: EvalContext): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.a

    if (aggregators.nonEmpty) {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val f = (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
        val aggregations = aggregators.map(_.zero)
        aggregatorA(0) = v
        aggregatorA(1) = va
        (gs, localSamplesBc.value, localAnnotationsBc.value).zipped
          .foreach {
            case (g, s, sa) =>
              aggregatorA(2) = s
              aggregatorA(3) = sa
              aggregators.iterator.zipWithIndex
                .foreach {
                  case (agg, i) =>
                    aggregations(i) = agg.seqOp(g, aggregations(i))
                }
          }
        aggregations.iterator
          .zip(aggregators.iterator)
          .foreach { case (res, agg) =>
            aggregatorA(agg.idx) = res
          }
      }
      Some(f)
    } else None
  }

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext): Option[(String) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.a

    if (aggregators.isEmpty)
      None
    else {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val nAggregations = aggregators.length
      val nSamples = vds.nSamples

      val baseArray = MultiArray2.fill[Any](nSamples, nAggregations)(null)
      for (i <- 0 until nSamples; j <- 0 until nAggregations) {
        baseArray.update(i, j, aggregators(j).zero)
      }

      val result = vds.rdd.treeAggregate(baseArray)({ case (arr, (v, (va, gs))) =>
        aggregatorA(0) = v
        aggregatorA(1) = va
        gs.iterator
          .zipWithIndex
          .foreach { case (g, i) =>
            aggregatorA(2) = localSamplesBc.value(i)
            aggregatorA(3) = localAnnotationsBc.value(i)

            for (j <- 0 until nAggregations) {
              arr.update(i, j, aggregators(j).seqOp(g, arr(i, j)))
            }
          }

        arr
      }, { case (arr1, arr2) =>
        for (i <- 0 until nSamples; j <- 0 until nAggregations) {
          arr1.update(i, j, aggregators(j).combOp(arr1(i, j), arr2(i, j)))
        }
        arr1
      }
      )

      val sampleIndex = vds.sampleIds.zipWithIndex.toMap
      Some((s: String) => {
        val i = sampleIndex(s)
        for (j <- 0 until nAggregations) {
          aggregatorA(aggregators(j).idx) = result(i, j)
        }
      })
    }
  }

  def makeFunctions(ec: EvalContext): (Array[Any], (Array[Any], (Any, Any)) => Array[Any],
    (Array[Any], Array[Any]) => Array[Any], (Array[Any]) => Unit) = {

    val aggregators = ec.aggregationFunctions.toArray

    val arr = ec.a

    val seqOp: (Array[Any], (Any, Any)) => Array[Any] = (array: Array[Any], b) => {
      val (aggT, annotation) = b
      ec.set(0, annotation)
      for (i <- array.indices) {
        array(i) = aggregators(i).seqOp(aggT, array(i))
      }
      array
    }
    val combOp: (Array[Any], Array[Any]) => Array[Any] = (arr1, arr2) => {
      for (i <- arr1.indices) {
        arr1(i) = aggregators(i).combOp(arr1(i), arr2(i))
      }
      arr1
    }

    val resultOp = (array: Array[Any]) => array.iterator
      .zip(aggregators.iterator)
      .foreach {
        case (res, agg) =>
          arr(agg.idx) = res
      }

    (aggregators.map(_.zero), seqOp, combOp, resultOp)
  }

}

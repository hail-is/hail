package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.MultiArray2
import org.broadinstitute.hail.variant._

object Aggregators {

  def buildVariantaggregations(vds: VariantDataset, ec: EvalContext): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.a

    if (aggregators.nonEmpty) {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val seqOps = aggregators.map(_._2)
      val combOps = aggregators.map(_._3)
      val endIndices = aggregators.map(_._4)
      val f = (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
        val aggregations = aggregators.map(_._1())
        aggregatorA(0) = v
        aggregatorA(1) = va
        (gs, localSamplesBc.value, localAnnotationsBc.value).zipped
          .foreach {
            case (g, s, sa) =>
              aggregatorA(2) = s
              aggregatorA(3) = sa
              aggregatorA(4) = g
              seqOps.iterator.zipWithIndex
                .foreach {
                  case (so, i) =>
                    aggregations(i) = so(aggregations(i))
                }
          }
        aggregations.iterator
          .zip(endIndices.iterator)
          .foreach { case (res, i) =>
            aggregatorA(i) = res
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

      val zeroVals = aggregators.map(_._1)
      val seqOps = aggregators.map(_._2)
      val combOps = aggregators.map(_._3)
      val endIndices = aggregators.map(_._4)

      val nAggregations = aggregators.length
      val nSamples = vds.nSamples

      val baseArray = MultiArray2.fill[Any](nSamples, nAggregations)(null)
      for (i <- 0 until nSamples; j <- 0 until nAggregations) {
        baseArray.update(i, j, zeroVals(j)())
      }

      val result = vds.rdd.treeAggregate(baseArray)({ case (arr, (v, va, gs)) =>
        aggregatorA(0) = v
        aggregatorA(1) = va
        gs.iterator
          .zipWithIndex
          .foreach { case (g, i) =>
            aggregatorA(2) = localSamplesBc.value(i)
            aggregatorA(3) = localAnnotationsBc.value(i)
            aggregatorA(4) = g

            for (j <- 0 until nAggregations) {
              arr.update(i, j, seqOps(j)(arr(i, j)))
            }
          }

        arr
      }, { case (arr1, arr2) =>
        for (i <- 0 until nSamples; j <- 0 until nAggregations) {
          arr1.update(i, j, combOps(j)(arr1(i, j), arr2(i, j)))
        }
        arr1
      }
      )

      val indices = aggregators.map(_._4)
      val sampleIndex = vds.sampleIds.zipWithIndex.toMap
      Some((s: String) => {
        val i = sampleIndex(s)
        for (j <- 0 until nAggregations) {
          aggregatorA(indices(j)) = result(i, j)
        }
      })
    }
  }

  def makeFunctions(ec: EvalContext): (Array[Any], (Array[Any], (Any, Any)) => Array[Any],
    (Array[Any], Array[Any]) => Array[Any], (Array[Any]) => Unit) = {

    val agg = ec.aggregationFunctions.toArray

    val arr = ec.a
    val zVals = agg.map(_._1.apply())
    val seqOps = agg.map(_._2)
    val combOps = agg.map(_._3)
    val indices = agg.map(_._4)

    val seqOp: (Array[Any], (Any, Any)) => Array[Any] = (array: Array[Any], b) => {
      ec.setAll(b._1, b._2)
      for (i <- array.indices) {
        val seqOp = seqOps(i)
        array(i) = seqOp(array(i))
      }
      array
    }
    val combOp: (Array[Any], Array[Any]) => Array[Any] = (arr1, arr2) => {
      for (i <- arr1.indices) {
        val combOp = combOps(i)
        arr1(i) = combOp(arr1(i), arr2(i))
      }
      arr1
    }

    val resultOp = (array: Array[Any]) => array.iterator
      .zip(indices.iterator)
      .foreach {
        case (res, index) =>
          arr(index) = res
      }

    (zVals, seqOp, combOp, resultOp)
  }

}

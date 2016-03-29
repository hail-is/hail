package org.broadinstitute.hail.methods

import org.broadinstitute.hail.expr.EvalContext
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations.Annotation

object Aggregators {

  def buildVariantaggregations(vds: VariantDataset, ec: EvalContext): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.aggregatorA

    if (aggregators.nonEmpty) {
      val sampleInfoBc = vds.sparkContext.broadcast(
        vds.localSamples.map(vds.sampleAnnotations)
          .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))

      val f = (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
        val aggregations = aggregators.map(_._1())
        gs.iterator
          .zip(sampleInfoBc.value.iterator)
          .foreach {
            case (g, (sa, s)) =>
              aggregatorA(0) = v
              aggregatorA(1) = va
              aggregatorA(2) = s
              aggregatorA(3) = sa
              aggregatorA(4) = g
              aggregators.iterator.zipWithIndex
                .foreach {
                  case ((zv, so, co), i) =>
                    aggregations(i) = so(aggregations(i))
                }
          }
        aggregations.iterator.zipWithIndex
          .foreach { case (res, i) =>
            aggregatorA(5 + i) = res
          }
      }
      Some(f)
    } else None
  }

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext): Option[Array[Array[Any]]] = {
    val aggregators = ec.aggregationFunctions
    val aggregatorA = ec.aggregatorA

    if (aggregators.isEmpty)
      None
    else {
      val aggregatorInternalArray = aggregators.toArray
      val sampleInfoBc = vds.sparkContext.broadcast(vds.localSamples
        .map(vds.sampleIds)
        .map(Sample)
        .zip(vds.localSamples.map(vds.sampleAnnotations)))
      val arr = vds.rdd.aggregate(Array.fill[Array[Any]](vds.nLocalSamples)(aggregatorInternalArray.map(_._1())))({ case (arr, (v, va, gs)) =>
        gs.iterator
          .zipWithIndex
          .foreach { case (g, i) =>
            aggregatorA(0) = v
            aggregatorA(1) = va
            aggregatorA(2) = sampleInfoBc.value(i)._1
            aggregatorA(3) = sampleInfoBc.value(i)._2
            aggregatorA(4) = g

            aggregatorInternalArray.iterator
              .zipWithIndex
              .foreach { case ((zv, seqOp, combOp), j) =>
                val iArray = arr(i)
                iArray(j) = seqOp(iArray(j))
              }
          }
        arr
      }, { case (arr1, arr2) =>
        val combOp = aggregatorInternalArray.map(_._3)
        arr1.iterator
          .zip(arr2.iterator)
          .map { case (ai1, ai2) =>
            ai1.iterator
              .zip(ai2.iterator)
              .zip(combOp.iterator)
              .map { case ((ij1, ij2), c) => c(ij1, ij2) }
              .toArray
          }
          .toArray
      })
      Some(arr)
    }
  }

}

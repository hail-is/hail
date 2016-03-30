package org.broadinstitute.hail.methods

import org.broadinstitute.hail.expr.EvalContext
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations.Annotation

object Aggregators {

  def buildVariantaggregations(vds: VariantDataset, ec: EvalContext,
    key: String): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {
    val aggregators = ec.children(key).aggregationFunctions.toArray
    val aggregatorA = ec.children(key).a

    if (aggregators.nonEmpty) {
      val sampleInfoBc = vds.sparkContext.broadcast(
        vds.localSamples.map(vds.sampleAnnotations)
          .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))

      val seqOps = aggregators.map(_._2)
      val combOps = aggregators.map(_._3)
      val endIndices = aggregators.map(_._4)
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

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext, key: String): Option[Array[Array[Any]]] = {
    val aggregators = ec.children(key).aggregationFunctions
    val aggregatorA = ec.children(key).a

    if (aggregators.isEmpty)
      None
    else {
      val aggregatorInternalArray = aggregators.toArray
      val sampleInfoBc = vds.sparkContext.broadcast(vds.localSamples
        .map(vds.sampleIds)
        .map(Sample)
        .zip(vds.localSamples.map(vds.sampleAnnotations)))

      val seqOps = aggregators.map(_._2)
      val combOps = aggregators.map(_._3)
      val endIndices = aggregators.map(_._4)

      val arr = vds.rdd.treeAggregate(Array.fill[Array[Any]](vds.nLocalSamples)(aggregatorInternalArray.map(_._1())))({
        case (arr, (v, va, gs)) =>
          gs.iterator
            .zipWithIndex
            .foreach { case (g, i) =>
              aggregatorA(0) = v
              aggregatorA(1) = va
              aggregatorA(2) = sampleInfoBc.value(i)._1
              aggregatorA(3) = sampleInfoBc.value(i)._2
              aggregatorA(4) = g

              seqOps.iterator
                .zipWithIndex
                .foreach { case (seqOp, j) =>
                  val iArray = arr(i)
                  iArray(j) = seqOp(iArray(j))
                }
            }
          arr
      }, { case (arr1, arr2) =>
        arr1.iterator
          .zip(arr2.iterator)
          .map { case (ai1, ai2) =>
            ai1.iterator
              .zip(ai2.iterator)
              .zip(combOps.iterator)
              .map { case ((ij1, ij2), c) => c(ij1, ij2) }
              .toArray
          }
          .toArray
      })
      Some(arr)
    }
  }

}

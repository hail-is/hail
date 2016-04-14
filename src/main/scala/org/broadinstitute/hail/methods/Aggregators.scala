package org.broadinstitute.hail.methods

import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations.Annotation

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
        (gs, localSamplesBc.value, localAnnotationsBc.value).zipped
          .foreach {
            case (g, s, sa) =>
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

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext): Option[(String) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.a

    if (aggregators.isEmpty)
      None
    else {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val seqOps = aggregators.map(_._2)
      val combOps = aggregators.map(_._3)
      val endIndices = aggregators.map(_._4)

      val functionMap = vds.sampleIds.zip(vds.rdd.treeAggregate(Array.fill[Array[Any]](vds.nSamples)(aggregators.map(_._1())))({
        case (arr, (v, va, gs)) =>
          gs.iterator
            .zipWithIndex
            .foreach { case (g, i) =>
              aggregatorA(0) = v
              aggregatorA(1) = va
              aggregatorA(2) = localSamplesBc.value(i)
              aggregatorA(3) = localAnnotationsBc.value(i)
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
      })).toMap

      val indices = aggregators.map(_._4)
      Some((s: String) => {
        functionMap(s).iterator
          .zip(indices.iterator)
          .foreach { case (value, j) =>
            aggregatorA(j) = value
          }
      })
    }
  }

}

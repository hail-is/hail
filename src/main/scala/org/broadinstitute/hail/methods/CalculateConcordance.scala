package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.sparkextras.OrderedRDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._

import Numeric.Implicits._

object ConcordanceCombiner {
  val schema = TArray(TArray(TLong))
}

class ConcordanceCombiner extends Serializable {
  // 5x5 square matrix indexed by [NoData, NoCall, HomRef, Het, HomVar] on each axis
  val mapping = MultiArray2.fill(5, 5)(0L)

  def mergeBoth(left: Int, right: Int) {
    mapping(left + 2, right + 2) += 1
  }

  def mergeLeft(left: Int) {
    mapping(left + 2, 0) += 1
  }

  def mergeRight(right: Int) {
    mapping(0, right + 2) += 1
  }

  def merge(other: ConcordanceCombiner): ConcordanceCombiner = {
    mapping.addElementWise(other.mapping)
    this
  }

  def reset() {
    val a = mapping.toArray
    var i = 0
    while (i < 25) {
      a(i) = 0L
      i += 1
    }
  }

  def report() {
    val innerTotal = (1 until 5).map(i => (1 until 5).map(j => mapping(i, j)).sum).sum
    val innerDiagonal = (1 until 5).map(i => mapping(i, i)).sum
    val total = mapping.sum
    info(
      s"""Summary of inner join concordance:
          |  Total observations: $innerTotal
          |  Total concordant observations: $innerDiagonal
          |  Total concordance: ${ (innerDiagonal.toDouble / innerTotal * 100).formatted("%.2f") }%""".stripMargin)
  }

  def toAnnotation = (0 until 5).map(i => (0 until 5).map(j => mapping(i, j)))
}

object CalculateConcordance {

  val schema = TStruct("concordance" -> ConcordanceCombiner.schema)

  def apply(left: VariantDataset, right: VariantDataset): (VariantDataset, VariantDataset) = {
    require(left.wasSplit && right.wasSplit, "passed unsplit dataset to Concordance")
    val overlap = left.sampleIds.toSet.intersect(right.sampleIds.toSet)
    if (overlap.isEmpty)
      fatal("No overlapping samples between datasets")

    info(
      s"""Found ${ overlap.size } overlapping samples
          |  Left: ${ left.nSamples } total samples
          |  Right: ${ right.nSamples } total samples""".stripMargin)

    val leftFiltered = left.filterSamples { case (s, _) => overlap(s) }
    val rightFiltered = right.filterSamples { case (s, _) => overlap(s) }

    val leftIds = leftFiltered.sampleIds
    val rightIds = rightFiltered.sampleIds

    assert(leftIds.toSet == overlap && rightIds.toSet == overlap)

    val leftIdIndex = leftIds.zipWithIndex.toMap
    val rightIdMapping = rightIds.map(leftIdIndex).toArray
    val rightIdMappingBc = left.sparkContext.broadcast(rightIdMapping)

    val join = leftFiltered.rdd.orderedOuterJoinDistinct(rightFiltered.rdd)

    val nSamples = leftIds.length
    val sampleResults = join.mapPartitions { it =>
      val arr = Array.ofDim[Int](nSamples)
      val comb = Array.fill(nSamples)(new ConcordanceCombiner)
      val rightMapping = rightIdMappingBc.value

      it.foreach { case (v, (v1, v2)) =>
        ((v1, v2): @unchecked) match {

          case (Some((_, leftGS)), Some((_, rightGS))) =>
            var i = 0
            rightGS.foreach { g =>
              arr(rightMapping(i)) = g.unboxedGT
              i += 1
            }
            assert(i == nSamples)
            i = 0
            leftGS.foreach { g =>
              comb(i).mergeBoth(g.unboxedGT, arr(i))
              i += 1
            }
          case (None, Some((_, rightGS))) =>
            var i = 0
            rightGS.foreach { g =>
              comb(rightMapping(i)).mergeRight(g.unboxedGT)
              i += 1
            }
            assert(i == nSamples)
          case (Some((_, leftGS)), None) =>
            var i = 0
            leftGS.foreach { g =>
              comb(i).mergeLeft(g.unboxedGT)
              i += 1
            }
        }
      }
      Iterator(comb)
    }.treeReduce { case (arr1, arr2) =>
      arr1.indices.foreach { i => arr1(i).merge(arr2(i)) }
      arr1
    }

    val global = new ConcordanceCombiner
    sampleResults.foreach(global.merge)

    global.report()


    val samples = VariantSampleMatrix(VariantMetadata(leftIds,
      sampleAnnotations = sampleResults.map { comb =>
        val anno = Annotation(comb.toAnnotation)
        assert(schema.typeCheck(anno))
        anno
      },
      globalAnnotation = Annotation(global.toAnnotation),
      saSignature = schema,
      vaSignature = TStruct.empty,
      globalSignature = schema,
      wasSplit = true),
      OrderedRDD.empty[Locus, Variant, (Annotation, Iterable[Genotype])](left.sparkContext))

    val variantResults = join.mapPartitions({ it =>
      val arr = Array.ofDim[Int](nSamples)
      val comb = new ConcordanceCombiner
      val rightMapping = rightIdMappingBc.value

      it.map { case (v, (value1, value2)) =>
        comb.reset()
        ((value1, value2): @unchecked) match {
          case (Some((_, leftGS)), Some((_, rightGS))) =>
            var i = 0
            rightGS.foreach { g =>
              arr(rightMapping(i)) = g.unboxedGT
              i += 1
            }
            assert(i == nSamples)
            i = 0
            leftGS.foreach { g =>
              comb.mergeBoth(g.unboxedGT, arr(i))
              i += 1
            }
          case (None, Some((_, gs2))) =>
            gs2.foreach { g =>
              comb.mergeRight(g.unboxedGT)
            }
          case (Some((_, gs1)), None) =>
            gs1.foreach { g => comb.mergeLeft(g.unboxedGT) }
        }
        val va = Annotation(comb.toAnnotation)
        assert(schema.typeCheck(va))
        (v, (va, Iterable.empty[Genotype]))
      }
    }, preservesPartitioning = true).asOrderedRDD

    val variants = VariantSampleMatrix(VariantMetadata(IndexedSeq.empty[String],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      globalAnnotation = Annotation(global.toAnnotation),
      saSignature = TStruct.empty,
      vaSignature = schema,
      globalSignature = schema,
      wasSplit = true),
      variantResults)

    (samples, variants)
  }
}

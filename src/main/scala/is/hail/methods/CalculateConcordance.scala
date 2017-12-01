package is.hail.methods

import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

object ConcordanceCombiner {
  val schema = TArray(TArray(TInt64()))
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
    val a = mapping.array
    var i = 0
    while (i < 25) {
      a(i) = 0L
      i += 1
    }
  }

  def nDiscordant: Long = {
    var n = 0L
    for (i <- 2 to 4)
      for (j <- 2 to 4)
        if (i != j)
          n += mapping(i, j)
    n
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

  def toAnnotation: IndexedSeq[IndexedSeq[Long]] =
    (0 until 5).map(i => (0 until 5).map(j => mapping(i, j)).toArray: IndexedSeq[Long]).toArray[IndexedSeq[Long]]
}

object CalculateConcordance {

  def apply(left: VariantDataset, right: VariantDataset): (IndexedSeq[IndexedSeq[Long]], KeyTable, KeyTable) = {
    require(left.wasSplit && right.wasSplit, "passed unsplit dataset to Concordance")
    val overlap = left.sampleIds.toSet.intersect(right.sampleIds.toSet)
    if (overlap.isEmpty)
      fatal("No overlapping samples between datasets")

    if (left.vSignature != right.vSignature)
      fatal(s"""Cannot compute concordance for datasets with different reference genomes:
              |  left: ${left.vSignature.toPrettyString(compact = true)}
              |  right: ${right.vSignature.toPrettyString(compact = true)}""")

    info(
      s"""Found ${ overlap.size } overlapping samples
         |  Left: ${ left.nSamples } total samples
         |  Right: ${ right.nSamples } total samples""".stripMargin)

    val leftFiltered = left.filterSamples { case (s, _) => overlap(s) }
    val rightFiltered = right.filterSamples { case (s, _) => overlap(s) }

    val sampleSchema = TStruct(
      "s" -> TString(),
      "nDiscordant" -> TInt64(),
      "concordance" -> ConcordanceCombiner.schema
    )

    val variantSchema = TStruct(
      "v" -> left.vSignature,
      "nDiscordant" -> TInt64(),
      "concordance" -> ConcordanceCombiner.schema
    )

    val leftIds = leftFiltered.sampleIds
    val rightIds = rightFiltered.sampleIds

    assert(leftIds.toSet == overlap && rightIds.toSet == overlap)

    val leftIdIndex = leftIds.zipWithIndex.toMap
    val rightIdMapping = rightIds.map(leftIdIndex).toArray
    val rightIdMappingBc = left.sparkContext.broadcast(rightIdMapping)

    val join = leftFiltered.typedRDD[Locus, Variant].orderedOuterJoinDistinct(rightFiltered.typedRDD[Locus, Variant])

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
              arr(rightMapping(i)) = Genotype.unboxedGT(g)
              i += 1
            }
            assert(i == nSamples)
            i = 0
            leftGS.foreach { g =>
              comb(i).mergeBoth(Genotype.unboxedGT(g), arr(i))
              i += 1
            }
          case (None, Some((_, rightGS))) =>
            var i = 0
            rightGS.foreach { g =>
              comb(rightMapping(i)).mergeRight(Genotype.unboxedGT(g))
              i += 1
            }
            assert(i == nSamples)
          case (Some((_, leftGS)), None) =>
            var i = 0
            leftGS.foreach { g =>
              comb(i).mergeLeft(Genotype.unboxedGT(g))
              i += 1
            }
        }
      }
      Iterator(comb)
    }.treeReduce { case (arr1, arr2) =>
      arr1.indices.foreach { i => arr1(i).merge(arr2(i)) }
      arr1
    }

    val variantRDD = join.mapPartitions { it =>
      val arr = Array.ofDim[Int](nSamples)
      val comb = new ConcordanceCombiner
      val rightMapping = rightIdMappingBc.value

      it.map { case (v, (value1, value2)) =>
        comb.reset()
        ((value1, value2): @unchecked) match {
          case (Some((_, leftGS)), Some((_, rightGS))) =>
            var i = 0
            rightGS.foreach { g =>
              arr(rightMapping(i)) = Genotype.unboxedGT(g)
              i += 1
            }
            assert(i == nSamples)
            i = 0
            leftGS.foreach { g =>
              comb.mergeBoth(Genotype.unboxedGT(g), arr(i))
              i += 1
            }
          case (None, Some((_, gs2))) =>
            gs2.foreach { g =>
              comb.mergeRight(Genotype.unboxedGT(g))
            }
          case (Some((_, gs1)), None) =>
            gs1.foreach { g => comb.mergeLeft(Genotype.unboxedGT(g)) }
        }
        val r = Row(v, comb.nDiscordant, comb.toAnnotation)
        assert(variantSchema.typeCheck(r))
        r
      }
    }

    val global = new ConcordanceCombiner
    sampleResults.foreach(global.merge)

    global.report()

    val sampleRDD = left.hc.sc.parallelize(leftFiltered.sampleIds.zip(sampleResults)
      .map { case (id, comb) => Row(id, comb.nDiscordant, comb.toAnnotation) })

    val sampleKT = KeyTable(left.hc, sampleRDD, sampleSchema, Array("s"))

    val variantKT = KeyTable(left.hc, variantRDD, variantSchema, Array("v"))

    (global.toAnnotation, sampleKT, variantKT)
  }
}

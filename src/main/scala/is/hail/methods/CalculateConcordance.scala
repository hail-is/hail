package is.hail.methods

import is.hail.annotations.UnsafeRow
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

object ConcordanceCombiner {
  val schema = TArray(TArray(TInt64()))
}

class ConcordanceCombiner extends Serializable {
  // 5x5 square matrix indexed by [NoData, NoCall, HomRef, Het, HomVar] on each axis
  val mapping = MultiArray2.fill(5, 5)(0L)

  def merge(left: Int, right: Int) {
    mapping(left, right) += 1
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

  def apply(left: MatrixTable, right: MatrixTable): (IndexedSeq[IndexedSeq[Long]], Table, Table) = {
    left.requireUniqueSamples("concordance")
    right.requireUniqueSamples("concordance")

    val overlap = left.stringSampleIds.toSet.intersect(right.stringSampleIds.toSet)
    if (overlap.isEmpty)
      fatal("No overlapping samples between datasets")

    if (!left.rowKeyTypes.sameElements(right.rowKeyTypes))
      fatal(s"""Cannot compute concordance for datasets with different key types:
              |  left: ${ left.rowKeyTypes.map(_.toString).mkString(", ") }
              |  right: ${ right.rowKeyTypes.map(_.toString).mkString(", ") }""")

    info(
      s"""Found ${ overlap.size } overlapping samples
         |  Left: ${ left.numCols } total samples
         |  Right: ${ right.numCols } total samples""".stripMargin)

    val leftPreIds = left.stringSampleIds
    val rightPreIds = right.stringSampleIds
    val leftFiltered = left.filterSamples { case (_, i) => overlap(leftPreIds(i)) }
    val rightFiltered = right.filterSamples { case (_, i) => overlap(rightPreIds(i)) }

    val sampleSchema = TStruct(
      "s" -> TString(),
      "nDiscordant" -> TInt64(),
      "concordance" -> ConcordanceCombiner.schema
    )

    val variantSchema = TStruct(
      left.rowKey.zip(left.rowKeyTypes) ++
        Array("nDiscordant" -> TInt64(), "concordance" -> ConcordanceCombiner.schema): _*
    )

    val leftIds = leftFiltered.stringSampleIds
    val rightIds = rightFiltered.stringSampleIds

    assert(leftIds.toSet == overlap && rightIds.toSet == overlap)

    val rightIdIndex = rightIds.zipWithIndex.toMap
    val leftToRight = leftIds.map(rightIdIndex).toArray
    val leftToRightBc = left.sparkContext.broadcast(leftToRight)

    val join = leftFiltered.rvd.orderedZipJoin(rightFiltered.rvd)

    val leftRowType = leftFiltered.rvRowType
    val rightRowType = rightFiltered.rvRowType

    val nSamples = leftIds.length
    val sampleResults = join.mapPartitions { it =>
      val comb = Array.fill(nSamples)(new ConcordanceCombiner)

      val lview = HardCallView(leftRowType)
      val rview = HardCallView(rightRowType)

      it.foreach { jrv =>
        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight

        if (lrv != null)
          lview.setRegion(lrv)
        if (rrv != null)
          rview.setRegion(rrv)

        var li = 0
        while (li < nSamples) {
          if (lrv != null)
            lview.setGenotype(li)
          if (rrv != null)
            rview.setGenotype(leftToRightBc.value(li))
          comb(li).merge(
            if (lrv != null) {
              if (lview.hasGT)
                Call.unphasedDiploidGtIndex(lview.getGT) + 2
              else
                1
            } else
              0,
            if (rrv != null) {
              if (rview.hasGT)
                Call.unphasedDiploidGtIndex(rview.getGT) + 2
              else
                1
            } else
              0)
          li += 1
        }
      }
      Iterator(comb)
    }.treeReduce { case (arr1, arr2) =>
      arr1.indices.foreach { i => arr1(i).merge(arr2(i)) }
      arr1
    }

    val leftRowKeysF = left.rowKeysF
    val rightRowKeysF = right.rowKeysF
    val variantRDD = join.mapPartitions { it =>
      val comb = new ConcordanceCombiner

      val lur = new UnsafeRow(leftRowType)
      val rur = new UnsafeRow(rightRowType)
      val lview = HardCallView(leftRowType)
      val rview = HardCallView(rightRowType)

      it.map { jrv =>
        comb.reset()

        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight

        val rowKeys: Row =
          if (lrv != null) {
            lur.set(lrv)
            leftRowKeysF(lur)
          } else {
            rur.set(rrv)
            rightRowKeysF(rur)
          }

        if (lrv != null)
          lview.setRegion(lrv)
        if (rrv != null)
          rview.setRegion(rrv)

        var li = 0
        while (li < nSamples) {
          if (lrv != null)
            lview.setGenotype(li)
          if (rrv != null)
            rview.setGenotype(leftToRightBc.value(li))
          comb.merge(
            if (lrv != null) {
              if (lview.hasGT)
                Call.unphasedDiploidGtIndex(lview.getGT) + 2
              else
                1
            } else
              0,
            if (rrv != null) {
              if (rview.hasGT)
                Call.unphasedDiploidGtIndex(rview.getGT) + 2
              else
                1
            } else
              0)
          li += 1
        }
        val r = Row.fromSeq(rowKeys.toSeq ++ Array(comb.nDiscordant, comb.toAnnotation))
        assert(variantSchema.typeCheck(r))
        r
      }
    }

    val global = new ConcordanceCombiner
    sampleResults.foreach(global.merge)

    global.report()

    val sampleRDD = left.hc.sc.parallelize(leftFiltered.stringSampleIds.zip(sampleResults)
      .map { case (id, comb) => Row(id, comb.nDiscordant, comb.toAnnotation) })

    val sampleKT = Table(left.hc, sampleRDD, sampleSchema, left.colKey)

    val variantKT = Table(left.hc, variantRDD, variantSchema, left.rowKey)

    (global.toAnnotation, sampleKT, variantKT)
  }
}

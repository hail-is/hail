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
    val overlap = left.sampleIds.toSet.intersect(right.sampleIds.toSet)
    if (overlap.isEmpty)
      fatal("No overlapping samples between datasets")

    if (left.vSignature != right.vSignature)
      fatal(s"""Cannot compute concordance for datasets with different reference genomes:
              |  left: ${ left.vSignature.toPrettyString(compact = true) }
              |  right: ${ right.vSignature.toPrettyString(compact = true) }""")

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

    val rightIdIndex = rightIds.zipWithIndex.toMap
    val leftToRight = leftIds.map(rightIdIndex).toArray
    val leftToRightBc = left.sparkContext.broadcast(leftToRight)

    val join = leftFiltered.rdd2.orderedZipJoin(rightFiltered.rdd2)

    val leftRowType = leftFiltered.rowType
    val rightRowType = rightFiltered.rowType

    val nSamples = leftIds.length
    val sampleResults = join.mapPartitions { it =>
      val comb = Array.fill(nSamples)(new ConcordanceCombiner)
      val leftToRight = leftToRightBc.value

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
            rview.setGenotype(leftToRight(li))
          comb(li).merge(
            if (lrv != null) {
              if (lview.hasGT)
                lview.getGT + 2
              else
                1
            } else
              0,
            if (rrv != null) {
              if (rview.hasGT)
                rview.getGT + 2
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

    val variantRDD = join.mapPartitions { it =>
      val comb = new ConcordanceCombiner
      val rightToLeft = leftToRightBc.value

      val lur = new UnsafeRow(leftRowType)
      val rur = new UnsafeRow(rightRowType)
      val lview = HardCallView(leftRowType)
      val rview = HardCallView(rightRowType)

      it.map { jrv =>
        comb.reset()

        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight

        val v =
          if (lrv != null) {
            lur.set(lrv)
            lur.get(1)
          } else {
            rur.set(rrv)
            rur.get(1)
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
            rview.setGenotype(leftToRight(li))
          comb.merge(
            if (lrv != null) {
              if (lview.hasGT)
                lview.getGT + 2
              else
                1
            } else
              0,
            if (rrv != null) {
              if (rview.hasGT)
                rview.getGT + 2
              else
                1
            } else
              0)
          li += 1
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

    val sampleKT = Table(left.hc, sampleRDD, sampleSchema, Array("s"))

    val variantKT = Table(left.hc, variantRDD, variantSchema, Array("v"))

    (global.toAnnotation, sampleKT, variantKT)
  }
}

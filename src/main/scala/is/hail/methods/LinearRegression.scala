package is.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.variant._
import scala.collection.mutable

class LinRegBuilder(y: DenseVector[Double]) extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var row = 0
  private var sparseLength = 0 // length of rowsX and valsX (ArrayBuilder has no length), used to track missingRowIndices
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private var sumYMissing = 0.0

  def merge(g: Genotype): LinRegBuilder = {
    (g.unboxedGT: @unchecked) match {
      case 0 =>
      case 1 =>
        rowsX += row
        valsX += 1d
        sparseLength += 1
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case 2 =>
        rowsX += row
        valsX += 2d
        sparseLength += 1
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
        sumYMissing += y(row)
    }
    row += 1

    this
  }

  def stats(y: DenseVector[Double], n: Int, minAC: Int): Option[(SparseVector[Double], Double, Double)] = {
    require(minAC > 0)

    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = n - nMissing
    val allHet = sumX == nPresent && sumXX == nPresent
    val allHomVar = sumX == 2 * nPresent

    if (sumX < minAC || allHomVar || allHet)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      val x = new SparseVector[Double](rowsXArray, valsXArray, n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some((x, xx, xy))
    }
  }
}

object LinearRegression {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset, pathVA: List[String], completeSamples: IndexedSeq[String], y: DenseVector[Double], cov: DenseMatrix[Double], minAC: Int): VariantDataset = {
    require(cov.rows == y.size)
    require(completeSamples.size == y.size)

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${plural(k, "covariate")} including intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n samples with $k ${plural(k, "covariate")} including intercept...")

    val completeSamplesSet = completeSamples.toSet
    assert(completeSamplesSet.size == completeSamples.size)
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val (newVAS, inserter) = vds.insertVA(LinearRegression.`type`, pathVA)

    vds.mapAnnotations{ case (v, va, gs) =>
      val lrb = new LinRegBuilder(yBc.value)
      gs.iterator.zipWithIndex.foreach { case (g, i) => if (sampleMaskBc.value(i)) lrb.merge(g) }

      val linRegAnnot = lrb.stats(yBc.value, n, minAC).map { stats =>
        val (x, xx, xy) = stats

        val qtx = QtBc.value * x
        val qty = QtyBc.value
        val xxp: Double = xx - (qtx dot qtx)
        val xyp: Double = xy - (qtx dot qty)
        val yyp: Double = yypBc.value

        val b = xyp / xxp
        val se = math.sqrt((yyp / xxp - b * b) / d)
        val t = b / se
        val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))

        Annotation(b, se, t, p)
      }

      inserter(va, linRegAnnot)
    }.copy(vaSignature = newVAS)
  }
}
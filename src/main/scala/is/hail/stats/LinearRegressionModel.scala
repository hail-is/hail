package is.hail.stats

import breeze.linalg.{*, DenseMatrix, DenseVector, Matrix, Vector, diag, inv, sum}
import breeze.numerics.sqrt
import is.hail.annotations.Annotation
import is.hail.expr.{TArray, TFloat64, TStruct}
import net.sourceforge.jdistlib.T

case class LinearRegressionStats(b: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(b, se, t, p)
}

case class LinearRegressionStatsNew(b: Array[Array[Double]], se: Array[Array[Double]], t: Array[Array[Double]], p: Array[Array[Double]]) {
  def toAnnotation: Annotation = Annotation(
    b.map(a => a: IndexedSeq[_]): IndexedSeq[_],
    se.map(a => a: IndexedSeq[_]): IndexedSeq[_],
    t.map(a => a: IndexedSeq[_]): IndexedSeq[_],
    p.map(a => a: IndexedSeq[_]): IndexedSeq[_])
}

object LinearRegressionModel {
  def schema = TStruct(
    ("beta", TFloat64),
    ("se", TFloat64),
    ("tstat", TFloat64),
    ("pval", TFloat64))

  def schemaNew = TStruct(
    ("beta", TArray(TArray(TDouble))),
    ("se", TArray(TArray(TDouble))),
    ("tstat", TArray(TArray(TDouble))),
    ("pval", TArray(TArray(TDouble))))
  
  // fit one row, one x, one y
  def fit(x: Vector[Double], y: Vector[Double], yyp: Double, qt: Matrix[Double], qty: Vector[Double], d: Int): LinearRegressionStats = {
    val qtx = qt * x
    val xxp = (x dot x) - (qtx dot qtx)
    val xyp = (x dot y) - (qtx dot qty)

    val b = xyp / xxp
    val se = math.sqrt((yyp / xxp - b * b) / d)
    val t = b / se
    val p = 2 * T.cumulative(-math.abs(t), d, true, false)

    LinearRegressionStats(b, se, t, p)
  }
  
  // fit many rows, one x, many y
  def fitBlock(X: DenseMatrix[Double], y: DenseMatrix[Double], yyp: DenseVector[Double], qt: DenseMatrix[Double], qty: DenseMatrix[Double], d: Int, blockLength: Int): Array[LinearRegressionStatsNew] = {
    
    val dInv = 1.0 / d
    val qtx: DenseMatrix[Double] = qt * X
    val xxpRec: DenseVector[Double] = 1.0 / (X.t(*, ::).map(r => r dot r) - qtx.t(*, ::).map(r => r dot r))
    val ytx: DenseMatrix[Double] = y.t * X
    assert(ytx.rows == y.cols && ytx.cols == blockLength)
    
    val xyp: DenseMatrix[Double] = ytx - (qty.t * qtx)

    // re-use xyp
    val b = xyp
    var i = 0
    while (i < blockLength) {
      xyp(::, i) :*= xxpRec(i)
      i += 1
    }
    val se = sqrt(dInv * (yyp * xxpRec.t - (b :* b)))
    val t = b :/ se
    val p = t.map(s => 2 * T.cumulative(-math.abs(s), d, true, false))
    
    Array.tabulate(blockLength)(j =>
      LinearRegressionStatsNew(
        Array(b(::, j).toArray),
        Array(se(::, j).toArray),
        Array(t(::, j).toArray),
        Array(p(::, j).toArray)))
  }
  
  // fit one row, many x, many y
  def fit(x: DenseMatrix[Double], y: DenseMatrix[Double], yyp: DenseVector[Double], qt: DenseMatrix[Double], qty: DenseMatrix[Double], d: Int): Option[LinearRegressionStatsNew] = {
    try {
      val nxs = x.cols
      val dInv = 1.0 / d
      val qtx = qt * x
      val xxpInv = inv((x.t * x) - (qtx.t * qtx))
      val xyp = (x.t * y) - (qtx.t * qty)
      
      val b = xxpInv * xyp
      val xypb = xyp :* b // FIXME can reuse xyp
      val se = sqrt(diag(xxpInv)).asDenseMatrix.t * sqrt(dInv * (yyp.asDenseMatrix - sum(xypb(::, *)))) // move densematrix up
      val t = b :/ se
      val p = t.map(stat => 2 * T.cumulative(-math.abs(stat), d, true, false))

      Some(LinearRegressionStatsNew(
        Array.tabulate(nxs)(i => b(i to i, ::).toArray),
        Array.tabulate(nxs)(i => se(i to i, ::).toArray),
        Array.tabulate(nxs)(i => t(i to i, ::).toArray),
        Array.tabulate(nxs)(i => p(i to i, ::).toArray)))
     } catch {
      case e: breeze.linalg.MatrixSingularException =>
        None
    }
  }
}

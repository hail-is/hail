package is.hail.stats

import breeze.linalg._
import breeze.numerics.sqrt
import is.hail.annotations.Annotation
import is.hail.expr.{TArray, TDouble, TStruct}
import net.sourceforge.jdistlib.T

case class LinearRegressionStats(b: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(b, se, t, p)
}

case class LinearRegressionMultiFieldStats(b: Array[Double], se: Array[Double], t: Array[Double], p: Array[Double]) {
  def toAnnotation: Annotation = Annotation(b: IndexedSeq[Double], se: IndexedSeq[Double], t: IndexedSeq[Double], p: IndexedSeq[Double])
}

case class LinearRegressionMultiPhenoStats(b: Array[Double], se: Array[Double], t: Array[Double], p: Array[Double]) {
  def toAnnotation: Annotation = Annotation(b: IndexedSeq[Double], se: IndexedSeq[Double], t: IndexedSeq[Double], p: IndexedSeq[Double])
}

object LinearRegressionModel {
  def schema = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

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
  
  def schemaMultiField: TStruct = schemaMultiPheno
  
  def fit(X: DenseMatrix[Double], y: Vector[Double], yyp: Double, qt: DenseMatrix[Double], qty: Vector[Double], d: Int): LinearRegressionMultiFieldStats = {
    val qtx = qt * X
    val xxp = (X.t * X) - (qtx.t * qtx)
    val xyp = (X.t * y) - (qtx.t * qty)
    
    val dRec = 1.0 / d
    
    try {
      val b = xxp \ xyp
      val se = sqrt(dRec * (yyp - (xyp dot b)) * diag(inv(xxp)))
      val t = b :/ se
      val p = t.map(stat => 2 * T.cumulative(-math.abs(stat), d, true, false))
      
      LinearRegressionMultiFieldStats(b.toArray, se.toArray, t.toArray, p.toArray)
    } catch {
      case e: breeze.linalg.MatrixSingularException =>
        LinearRegressionMultiFieldStats(null, null, null, null)
    }
  }

  def schemaMultiPheno = TStruct(
    ("beta", TArray(TDouble)),
    ("se", TArray(TDouble)),
    ("tstat", TArray(TDouble)),
    ("pval", TArray(TDouble)))

  def fitMultiPheno(x: Vector[Double], y: DenseMatrix[Double], yyp: DenseVector[Double], qt: Matrix[Double],
    qty: DenseMatrix[Double], d: Int): LinearRegressionMultiPhenoStats = {
    
    val qtx = qt * x
    val xxpRec = 1 / ((x dot x) - (qtx dot qtx))
    val xyp: Vector[Double] = (y.t * x) - (qty.t * qtx)

    val dRec = 1.0 / d
    val b = xxpRec * xyp
    val se = sqrt(dRec * (xxpRec * yyp - (b :* b)))
    val t = b :/ se
    val p = t.map(stat => 2 * T.cumulative(-math.abs(stat), d, true, false))
    
    LinearRegressionMultiPhenoStats(b.toArray, se.toArray, t.toArray, p.toArray)
  }
}

package is.hail.stats

import breeze.linalg.{Matrix, Vector}
import is.hail.annotations.Annotation
import is.hail.expr.{TDouble, TStruct}
import net.sourceforge.jdistlib.T

case class LinearRegressionStats(b: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(b, se, t, p)
}

object LinearRegressionModel {
    val schema = TStruct(
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
}

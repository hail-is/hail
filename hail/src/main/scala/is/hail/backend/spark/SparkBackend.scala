package is.hail.backend.spark

import is.hail.HailContext
import is.hail.annotations.{Region, SafeRow}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.expr.types.physical.PTuple
import is.hail.utils._
import org.apache.spark.SparkContext
import org.json4s.jackson.JsonMethods

object SparkBackend {
  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val value = execute(HailContext.get.sc, ir)
    JsonMethods.compact(
      JSONAnnotationImpex.exportAnnotation(value, t))
  }

  def execute(sc: SparkContext, ir0: IR, optimize: Boolean = true): Any = {
    var ir = ir0

    println("SparkBackend.execute got", Pretty(ir))

    ir = ir.unwrap
    if (optimize)
      ir = Optimize(ir, noisy = true, canGenerateLiterals = true)
    ir = LiftLiterals(ir).asInstanceOf[IR]
    ir = LowerMatrixIR(ir)
    if (optimize)
      ir = Optimize(ir, noisy = true, canGenerateLiterals = false)

    println("SparkBackend.execute to lower", Pretty(ir))

    val pipeline = LowerTableIR.lower(ir)
    Region.scoped { region =>
      val (t, off) = pipeline.execute(sc, region)
      SafeRow(t, region, off).get(0)
    }
  }
}

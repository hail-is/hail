package is.hail.backend.spark

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import org.json4s.jackson.JsonMethods

object SparkBackend {
  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val value = execute(ir)
    JsonMethods.compact(
      JSONAnnotationImpex.exportAnnotation(value, t))
  }

  def execute(ir0: IR): Any = {
    var ir = ir0

    println("SparkBackend.execute got", Pretty(ir))

    ir = ir.unwrap
    ir = Optimize(ir, noisy = true, canGenerateLiterals = true)
    ir = LiftLiterals(ir).asInstanceOf[IR]
    ir = LowerMatrixIR(ir)
    ir = Optimize(ir, noisy = true, canGenerateLiterals = false)

    println("LocalBackend.execute to lower", Pretty(ir))

    ir = LowerTableIR.lower(ir)

    println("LocalBackend.execute lowered", Pretty(ir))

    ir = Optimize(ir, noisy = true, canGenerateLiterals = false)

    println("LocalBackend.execute", Pretty(ir))

    Interpret[Any](ir)
  }
}

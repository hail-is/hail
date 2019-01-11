package is.hail.backend.spark

import is.hail.HailContext
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.expr.types.virtual._
import is.hail.utils._
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

    println("SparkBackend.execute to lower", Pretty(ir))

    val SparkCollect(stages, value) = LowerTableIR.lower(ir)
    val bindings = stages.mapValues(stage => (stage.execute(HailContext.get.sc).collect().toFastIndexedSeq, TArray(stage.body.typ)))

    Interpret[Any](value, Env[(Any, Type)](bindings.toSeq: _*), FastIndexedSeq(), None)
  }
}

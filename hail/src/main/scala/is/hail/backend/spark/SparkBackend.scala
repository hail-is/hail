package is.hail.backend.spark

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.IR
import org.json4s.jackson.JsonMethods

object SparkBackend {
  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val value = execute(ir)
    JsonMethods.compact(
      JSONAnnotationImpex.exportAnnotation(value, t))
  }

  def execute(ir: IR): Any = {
    ???
  }
}

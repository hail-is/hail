package is.hail.backend.local

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.utils.{ExecutionTimer, Timings}
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

object LocalBackend {
  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val (value, timings) = execute(ir)
    val jsonValue = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    timings.logInfo()

    Serialization.write(Map("value" -> jsonValue, "timings" -> timings.value))(new DefaultFormats {})
  }

  def execute(ir0: IR): (Any, Timings) = {
    val timer = new ExecutionTimer("Just Interpret")
    var ir = ir0

    println(("LocalBackend.execute got", Pretty(ir)))

    ir = ir.unwrap
    ir = timer.time(
      Optimize(ir, noisy = true, canGenerateLiterals = true, context = Some(s"LocalBackend.execute - first pass")),
      "optimize first pass")
    ir = timer.time(LiftNonCompilable(ir).asInstanceOf[IR], "lift non-compilable")
    ir = timer.time(LowerMatrixIR(ir), "lower MatrixIR")
    ir = timer.time(
      Optimize(ir, noisy = true, canGenerateLiterals = false, context = Some("LocalBackend.execute - after MatrixIR lowering")),
      "optimize after matrix lowering")

    println(("LocalBackend.execute to lower", Pretty(ir)))

    ir = timer.time(LowerTableIR.lower(ir), "lowering TableIR")

    println(("LocalBackend.execute lowered", Pretty(ir)))

    ir = timer.time(
      Optimize(ir, noisy = true, canGenerateLiterals = false, context = Some("LocalBackend.execute - after TableIR lowering")),
      "optimize after table lowering")

    println(("LocalBackend.execute", Pretty(ir)))

    val value = timer.time(Interpret[Any](ir), "runtime")
    (value, timer.timings)
  }
}

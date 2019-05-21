package is.hail.backend.spark

import is.hail.{HailContext, cxx}
import is.hail.annotations.{Region, SafeRow}
import is.hail.cxx.CXXUnsupportedOperation
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual.TVoid
import is.hail.utils._
import org.apache.spark.SparkContext
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

object SparkBackend {
  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val (value, timings) = execute(ir)
    val jsonValue = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    timings.logInfo()

    Serialization.write(Map("value" -> jsonValue, "timings" -> timings.value))(new DefaultFormats {})
  }

  def cxxExecute(sc: SparkContext, ir0: IR, optimize: Boolean = true): (Any, Timings) = {
    val timer = new ExecutionTimer("CXX Compile")

    val ir = try {
      LowerTableIR(ir0, timer, optimize)
    } catch {
      case e: SparkBackendUnsupportedOperation =>
        throw new CXXUnsupportedOperation(s"Failed lowering step:\n${e.getMessage}")
    }

    val value = ir.typ match {
      case TVoid =>
        val f = timer.time(cxx.Compile(ir, optimize), "SparkBackend.execute - CXX compile")
        timer.time(Region.scoped { region => f(region.get()) }, "SparkBackend.execute - Runtime")
        Unit
      case _ =>
        val pipeline = MakeTuple(FastIndexedSeq(ir))
        val f = timer.time(cxx.Compile(pipeline, optimize: Boolean), "SparkBackend.execute - CXX compile")
        timer.time(
          Region.scoped { region =>
            val off = f(region.get())
            SafeRow(pipeline.pType.asInstanceOf[PTuple], region, off).get(0)
          },
          "SparkBackend.execute - Runtime")
    }

    (value, timer.timings)
  }

  def execute(ir: IR, optimize: Boolean = true): (Any, Timings) = {
    val hc = HailContext.get
    try {
      if (hc.flags.get("cpp") == null)
        throw new CXXUnsupportedOperation("'cpp' flag not enabled.")
      cxxExecute(hc.sc, ir, optimize)
    } catch {
      case (_: CXXUnsupportedOperation | _: SparkBackendUnsupportedOperation) =>
        CompileAndEvaluate(ir, optimize = optimize)
    }
  }
}

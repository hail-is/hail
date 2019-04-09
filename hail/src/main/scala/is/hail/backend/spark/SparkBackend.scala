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
import org.json4s.jackson.JsonMethods

object SparkBackend {
  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val res, times = execute(ir)
    JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(res, t))
  }

  def cxxExecute(sc: SparkContext, ir0: IR, optimize: Boolean = true): (Any, Map[String, Long]) = {
    val timer = new ExecutionTimer()
    var ir = ir0

    ir = ir.unwrap
    if (optimize) {
      val context = "SparkBackend.execute - first pass"
      ir = timer.time(Optimize(ir, noisy = true, canGenerateLiterals = true, Some(context)), context)
    }

    ir = timer.time(LiftNonCompilable(ir).asInstanceOf[IR], "SparkBackend.execute - lifting non-compilable")
    ir = timer.time(LowerMatrixIR(ir), "SparkBackend.execute - lowering MatrixIR")

    if (optimize) {
      val context = "SparkBackend.execute - after MatrixIR lowering"
      ir = timer.time(Optimize(ir, noisy = true, canGenerateLiterals = true, Some(context)), context)
    }

    val res = ir.typ match {
      case TVoid =>
        val f = timer.time(cxx.Compile(ir, optimize), "SparkBackend.execute - CXX compile")
        timer.time(Region.scoped { region => f(region.get()) }, "SparkBackend.execute - Runtime")
        Unit
      case _ =>
        val pipeline = MakeTuple(FastIndexedSeq(LowerTableIR.lower(ir)))
        val f = timer.time(cxx.Compile(pipeline, optimize: Boolean), "SparkBackend.execute - CXX compile")
        timer.time(
          Region.scoped { region =>
            val off = f(region.get())
            SafeRow(pipeline.pType.asInstanceOf[PTuple], region, off).get(0)
          },
          "SparkBackend.execute - Runtime")
    }
    
    (res, timer.times)
  }

  def execute(ir: IR, optimize: Boolean = true): (Any, Map[String, Long]) = {
    val hc = HailContext.get
    try {
      if (hc.flags.get("cpp") == null)
        throw new CXXUnsupportedOperation("'cpp' flag not enabled.")
      cxxExecute(hc.sc, ir, optimize)
    } catch {
      case _: CXXUnsupportedOperation =>
        CompileAndEvaluate(ir, optimize = optimize)
    }
  }
}

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
    val value = execute(ir)
    JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
  }

  def cxxExecute(sc: SparkContext, ir0: IR, optimize: Boolean = true): Any = {
    var ir = ir0

    ir = ir.unwrap
    if (optimize)
      ir = Optimize(ir, noisy = true, canGenerateLiterals = true, context = Some("SparkBackend.execute - first pass"))
    ir = LiftNonCompilable(ir).asInstanceOf[IR]
    ir = LowerMatrixIR(ir)
    if (optimize)
      ir = Optimize(ir, noisy = true, canGenerateLiterals = true, context = Some("SparkBackend.execute - after MatrixIR lowering"))

    ir.typ match {
      case TVoid =>
        val f = cxx.Compile(ir, optimize)
        Region.scoped { region => f(region.get()) }
        Unit
      case _ =>
        val pipeline = MakeTuple(FastIndexedSeq(LowerTableIR.lower(ir)))
        val f = cxx.Compile(pipeline, optimize)
        Region.scoped { region =>
          val off = f(region.get())
          SafeRow(pipeline.pType.asInstanceOf[PTuple], region, off).get(0)
        }
    }
  }

  def execute(ir: IR, optimize: Boolean = true): Any = {
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

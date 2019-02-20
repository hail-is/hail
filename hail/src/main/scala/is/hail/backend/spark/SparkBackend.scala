package is.hail.backend.spark

import is.hail.{HailContext, cxx}
import is.hail.annotations.{Region, SafeRow}
import is.hail.cxx.CXXUnsupportedOperation
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.expr.types.physical.{PInt32, PTuple}
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

  def executeOrError(sc: SparkContext, ir0: IR, optimize: Boolean = true): Any = {
    println("trying to execute IR")
    var ir = ir0

    ir = ir.unwrap
    if (optimize)
      ir = Optimize(ir, noisy = true, canGenerateLiterals = true, context = Some("SparkBackend.execute - first pass"))
    ir = LiftNonCompilable(ir).asInstanceOf[IR]
    ir = LowerMatrixIR(ir)
    if (optimize)
      ir = Optimize(ir, noisy = true, canGenerateLiterals = false, context = Some("SparkBackend.execute - after MatrixIR lowering"))

    val pipeline = MakeTuple(FastIndexedSeq(LowerTableIR.lower(ir)))
//    println(s"executing \n${ Pretty(pipeline) }")

    val f = cxx.Compile("foo", PTuple(FastSeq(PInt32())), pipeline, optimize: Boolean)

    println("executing IR")

    Region.scoped { region =>
      val off = f(region.get(), 0L)
      SafeRow(pipeline.pType.asInstanceOf[PTuple], region, off).get(0)
    }
  }

  def execute(sc: SparkContext, ir: IR, optimize: Boolean = true): Any = {
    try {
      executeOrError(sc, ir, optimize)
    } catch {
      case e: CXXUnsupportedOperation =>
        Interpret(ir, optimize = optimize)
    }
  }
}

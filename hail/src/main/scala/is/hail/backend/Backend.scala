package is.hail.backend

import java.io.PrintWriter

import is.hail.HailContext
import is.hail.annotations.{Region, SafeRow}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, LoweringPipeline}
import is.hail.expr.ir.{Compilable, Compile, CompileAndEvaluate, ExecuteContext, IR, MakeTuple, Pretty, TypeCheck}
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual.TVoid
import is.hail.utils._
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.reflect.ClassTag

abstract class BroadcastValue[T] { def value: T }

abstract class Backend {
  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U]

  def jvmLowerAndExecute(ir0: IR, optimize: Boolean, print: Option[PrintWriter] = None): (Any, ExecutionTimer) = {
    ExecuteContext.scoped { ctx =>

      val ir = LoweringPipeline.tableLowerer.apply(ctx, ir0, optimize).asInstanceOf[IR]

      if (!Compilable(ir))
        throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${ Pretty(ir) }")

      val res = ir.typ match {
        case TVoid =>
          val (_, f) = ctx.timer.time("Compile")(Compile[Unit](ctx, ir, print))
          ctx.timer.time("Run")(f(0, ctx.r)(ctx.r))

        case _ =>
          val (pt: PTuple, f) = ctx.timer.time("Compile")(Compile[Long](ctx, MakeTuple.ordered(FastSeq(ir)), print))
          ctx.timer.time("Run")(SafeRow(pt, ctx.r, f(0, ctx.r)(ctx.r)).get(0))
      }

      (res, ctx.timer)
    }
  }

  def execute(ir: IR, optimize: Boolean): (Any, ExecutionTimer) = {
    TypeCheck(ir)
    try {
      if (HailContext.get.flags.get("lower") == null)
        throw new LowererUnsupportedOperation("lowering not enabled")
      jvmLowerAndExecute(ir, optimize)
    } catch {
      case _: LowererUnsupportedOperation =>
        ExecuteContext.scoped(ctx => (CompileAndEvaluate(ctx, ir, optimize = optimize), ctx.timer))
    }
  }

  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val (value, timings) = execute(ir, optimize = true)
    val jsonValue = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    timings.finish()
    timings.logInfo()

    Serialization.write(Map("value" -> jsonValue, "timings" -> timings.asMap()))(new DefaultFormats {})
  }

  def asSpark(): SparkBackend = fatal("SparkBackend needed for this operation.")
}

package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.{Region, SafeRow}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.{Compilable, Compile, CompileAndEvaluate, ExecuteContext, IR, IRParser, MakeTuple, Pretty, TypeCheck}
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual.TVoid
import is.hail.io.CodecSpec
import is.hail.utils._
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.reflect.ClassTag

abstract class BroadcastValue[T] { def value: T }

abstract class Backend {
  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U]

  def lower(ir: IR, timer: Option[ExecutionTimer], optimize: Boolean = true): IR =
      LowerTableIR(ir, timer, optimize)

  def jvmLowerAndExecute(ir0: IR, optimize: Boolean): (Any, Timings) = {
    val timer = new ExecutionTimer("Backend.execute")
    val ir = lower(ir0, Some(timer), optimize)

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ir)}")

    val res = Region.scoped { region =>
      ir.typ match {
        case TVoid =>
          val (_, f) = timer.time(Compile[Unit](ir), "JVM compile")
          timer.time(f(0, region)(region), "Runtime")
        case _ =>
          val (pt: PTuple, f) = timer.time(Compile[Long](MakeTuple.ordered(FastSeq(ir))), "JVM compile")
          timer.time(SafeRow(pt, region, f(0, region)(region)).get(0), "Runtime")
      }
    }

    (res, timer.timings)
  }

  def execute(ir: IR, optimize: Boolean): (Any, Timings) = {
    TypeCheck(ir)
    try {
      if (HailContext.get.flags.get("lower") == null)
        throw new LowererUnsupportedOperation("lowering not enabled")
      jvmLowerAndExecute(ir, optimize)
    } catch {
      case _: LowererUnsupportedOperation =>
        ExecuteContext.scoped(ctx => CompileAndEvaluate(ctx, ir, optimize = optimize))
    }
  }

  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val (value, timings) = execute(ir, optimize = true)
    val jsonValue = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    timings.logInfo()

    Serialization.write(Map("value" -> jsonValue, "timings" -> timings.value))(new DefaultFormats {})
  }

  def encode(ir0: IR, codecString: String): (String, Array[Byte]) = {
    val codec = CodecSpec.fromShortString(codecString)
    val ir = lower(ir0, None, false)

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ir)}")
    if (ir.typ == TVoid)
      throw new LowererUnsupportedOperation(s"lowered to unit-valued IR: ${Pretty(ir)}")

    Region.scoped { region =>
      val (pt: PTuple, f) = Compile[Long](MakeTuple.ordered(FastSeq(ir)))
      val enc = codec.makeCodecSpec2(pt)
      (pt.parsableString(), enc.encode(pt, region, f(0, region)(region)))
    }
  }

  def asSpark(): SparkBackend = fatal("SparkBackend needed for this operation.")
}

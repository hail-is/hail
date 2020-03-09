package is.hail.backend

import is.hail.annotations.UnsafeRow
import is.hail.expr.ir.IRParser
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical.{PBaseStruct, PType}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import java.io.{ByteArrayInputStream, PrintWriter}

import is.hail.HailContext
import is.hail.annotations.{Region, SafeRow}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LowererUnsupportedOperation, LoweringPipeline}
import is.hail.expr.ir.{BlockMatrixIR, Compilable, Compile, CompileAndEvaluate, ExecuteContext, IR, MakeTuple, Pretty, TypeCheck}
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual.TVoid
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.reflect.ClassTag

abstract class BroadcastValue[T] { def value: T }

abstract class ValueCache {
  def persistBlockMatrix(id: String, value: BlockMatrix, storageLevel: String): Unit
  def getPersistedBlockMatrix(id: String): BlockMatrix
  def unpersistBlockMatrix(id: String): Unit
}

abstract class Backend {
  def cache: ValueCache = asSpark().cache

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U]

  private[this] def executionResultToAnnotation(ctx: ExecuteContext, result: Either[Unit, (PTuple, Long)]) = result match {
    case Left(x) => x
    case Right((pt, off)) => SafeRow(pt, off).get(0)
  }

  def jvmLowerAndExecute(ir0: IR, optimize: Boolean, lowerTable: Boolean, lowerBM: Boolean, print: Option[PrintWriter] = None): (Any, ExecutionTimer) =
    ExecuteContext.scoped { ctx =>
      val (l, r) = _jvmLowerAndExecute(ctx, ir0, optimize, lowerTable, lowerBM, print)
      (executionResultToAnnotation(ctx, l), r)
    }

  private[this] def _jvmLowerAndExecute(ctx: ExecuteContext, ir0: IR, optimize: Boolean, lowerTable: Boolean, lowerBM: Boolean, print: Option[PrintWriter] = None): (Either[Unit, (PTuple, Long)], ExecutionTimer) = {
    val typesToLower: DArrayLowering.Type = (lowerTable, lowerBM) match {
      case (true, true) => DArrayLowering.All
      case (true, false) => DArrayLowering.TableOnly
      case (false, true) => DArrayLowering.BMOnly
      case (false, false) => throw new LowererUnsupportedOperation("no lowering enabled")
    }
    val ir = LoweringPipeline.darrayLowerer(typesToLower).apply(ctx, ir0, optimize).asInstanceOf[IR]

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${ Pretty(ir) }")

    val res = ir.typ match {
      case TVoid =>
        val (_, f) = ctx.timer.time("Compile")(Compile[Unit](ctx, ir, print))
        ctx.timer.time("Run")(Left(f(0, ctx.r)(ctx.r)))

      case _ =>
        val (pt: PTuple, f) = ctx.timer.time("Compile")(Compile[Long](ctx, MakeTuple.ordered(FastSeq(ir)), print))
        ctx.timer.time("Run")(Right((pt, f(0, ctx.r)(ctx.r))))
    }

    (res, ctx.timer)
  }

  def execute(ir: IR, optimize: Boolean): (Any, ExecutionTimer) =
    ExecuteContext.scoped { ctx =>
      val (l, r) = _execute(ctx, ir, optimize)
      (executionResultToAnnotation(ctx, l), r)
    }

  private[this] def _execute(ctx: ExecuteContext, ir: IR, optimize: Boolean): (Either[Unit, (PTuple, Long)], ExecutionTimer) = {
    TypeCheck(ir)
    try {
      val lowerTable = HailContext.get.flags.get("lower") != null
      val lowerBM = HailContext.get.flags.get("lower_bm") != null
      _jvmLowerAndExecute(ctx, ir, optimize, lowerTable, lowerBM)
    } catch {
      case _: LowererUnsupportedOperation =>
        (CompileAndEvaluate._apply(ctx, ir, optimize = optimize), ctx.timer)
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

  def encodeToBytes(ir: IR, bufferSpecString: String): (String, Array[Byte]) = {
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    ExecuteContext.scoped { ctx =>
      _execute(ctx, ir, true)._1 match {
        case Left(_) => throw new RuntimeException("expression returned void")
        case Right((t, off)) =>
          assert(t.size == 1)
          val elementType = t.fields(0).typ
          val codec = TypedCodecSpec(
            EType.defaultFromPType(elementType), elementType.virtualType, bs)
          assert(t.isFieldDefined(off, 0))
          (elementType.toString, codec.encode(elementType, ctx.r, t.loadField(off, 0)))
      }
    }
  }

  def decodeToJSON(ptypeString: String, b: Array[Byte], bufferSpecString: String): String = {
    val t = IRParser.parsePType(ptypeString)
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    val codec = TypedCodecSpec(EType.defaultFromPType(t), t.virtualType, bs)
    using(Region()) { r =>
      val (pt, off) = codec.decode(t.virtualType, b, r)
      assert(pt.virtualType == t.virtualType)
      JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(
        UnsafeRow.read(pt, r, off), pt.virtualType))
    }
  }

  def asSpark(): SparkBackend = fatal("SparkBackend needed for this operation.")
}

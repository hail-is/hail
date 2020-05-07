package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}

case class CodeCacheKey(aggSigs: IndexedSeq[AggStatePhysicalSignature], args: Seq[(String, PType)], body: IR)

case class CodeCacheValue(typ: PType, f: (Int, Region) => Any)

object Compile {
  private[this] val codeCache: Cache[CodeCacheKey, CodeCacheValue] = new Cache(50)

  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    params: IndexedSeq[(String, PType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]], expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true,
    print: Option[PrintWriter] = None
  ): (PType, (Int, Region) => F) = {

    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(FastIndexedSeq[AggStatePhysicalSignature](), params.map { case (n, pt) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => F])
      case None =>
    }

    var ir = body
    ir = Subst(ir, BindingEnv(params
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer.apply(ctx, ir, optimize).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv.empty)
    InferPType(ir, Env.empty[PType])
    val returnType = ir.pType

    val fb = EmitFunctionBuilder[F](ctx, "Compiled",
      CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) =>
        EmitParamType(pt)
      }, returnType)

    /*
    {
      def visit(x: IR): Unit = {
        println(f"${ System.identityHashCode(x) }%08x    ${ x.getClass.getSimpleName } ${ x.pType }")
        Children(x).foreach {
          case c: IR => visit(c)
        }
      }

      visit(ir)
    }
     */

    assert(fb.mb.parameterTypeInfo == expectedCodeParamTypes, s"expected $expectedCodeParamTypes, got ${ fb.mb.parameterTypeInfo }")
    assert(fb.mb.returnTypeInfo == expectedCodeReturnType, s"expected $expectedCodeReturnType, got ${ fb.mb.returnTypeInfo }")

    Emit(ctx, ir, fb)

    val f = fb.resultWithIndex(print)
    codeCache += k -> CodeCacheValue(ir.pType, f)

    (returnType, f)
  }
}

object CompileWithAggregators2 {
  private[this] val codeCache: Cache[CodeCacheKey, CodeCacheValue] = new Cache(50)

  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    aggSigs: Array[AggStatePhysicalSignature],
    params: IndexedSeq[(String, PType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]], expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true
  ): (PType, (Int, Region) => (F with FunctionWithAggRegion)) = {
    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(aggSigs, params.map { case (n, pt) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
      case None =>
    }

    var ir = body
    ir = Subst(ir, BindingEnv(params
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer.apply(ctx, ir, optimize).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv(Env.fromSeq[Type](params.map { case (name, t) => name -> t.virtualType })))

    InferPType(ir, Env.empty[PType], aggSigs, null, null)

    val returnType = ir.pType
    val fb = EmitFunctionBuilder[F](ctx, "CompiledWithAggs",
      CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) =>
        EmitParamType(pt)
      }, returnType)

    /*
    {
      def visit(x: IR): Unit = {
        println(f"${ System.identityHashCode(x) }%08x    ${ x.getClass.getSimpleName } ${ x.pType }")
        Children(x).foreach {
          case c: IR => visit(c)
        }
      }

      visit(ir)
    }
     */

    Emit(ctx, ir, fb, Some(aggSigs))

    val f = fb.resultWithIndex()
    codeCache += k -> CodeCacheValue(ir.pType, f)
    (ir.pType, f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
  }
}

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

  private def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    print: Option[PrintWriter],
    args: Seq[(String, PType, ClassTag[_])],
    argTypeInfo: Array[MaybeGenericTypeInfo[_]],
    body: IR,
    optimize: Boolean
  ): (PType, (Int, Region) => F) = {
    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(args.map { case (n, _, _) => n -> n }: _*))
    val k = CodeCacheKey(FastIndexedSeq[AggStatePhysicalSignature](), args.map { case (n, pt, _) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => F])
      case None =>
    }

    val fb = EmitFunctionBuilder[F]("Compiled", argTypeInfo, GenericTypeInfo[R]())

    var ir = body
    ir = Subst(ir, BindingEnv(args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer.apply(ctx, ir, optimize).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv.empty)

    InferPType(ir, Env(args.map { case (n, pt, _) => n -> pt}: _*))

    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])

    /*
    {
      def visit(x: IR): Unit = {
        println(f"${ System.identityHashCode(x) }%08x    ${ x.getClass.getSimpleName } ${ x.pType }")
        Children(x).foreach {
          case c: IR => visit(c)
        }
      }

      visit(ir)
    } */

    Emit(ctx, ir, fb)

    val f = fb.resultWithIndex(print)
    codeCache += k -> CodeCacheValue(ir.pType, f)

    (ir.pType, f)
  }

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    print: Option[PrintWriter],
    args: Seq[(String, PType, ClassTag[_])],
    body: IR,
    optimize: Boolean
  ): (PType, (Int, Region) => F) = {
    assert(args.forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t.virtualType) == ct })

    val ab = new ArrayBuilder[MaybeGenericTypeInfo[_]]()
    ab += GenericTypeInfo[Region]()
    args.foreach { case (_, t, _) =>
      ab += GenericTypeInfo()(typeToTypeInfo(t))
      ab += GenericTypeInfo[Boolean]()
    }

    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] = ab.result()

    Compile[F, R](ctx, print, args, argTypeInfo, body, optimize)
  }

  def apply[R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    body: IR,
    print: Option[PrintWriter],
    optimize: Boolean = true
  ): (PType, (Int, Region) => AsmFunction1[Region, R]) = {
    apply[AsmFunction1[Region, R], R](ctx, print, FastSeq[(String, PType, ClassTag[_])](), body, optimize)
  }

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    body: IR,
    optimize: Boolean,
    print: Option[PrintWriter]
  ): (PType, (Int, Region) => AsmFunction3[Region, T0, Boolean, R]) = {
    apply[AsmFunction3[Region, T0, Boolean, R], R](ctx, print, FastSeq((name0, typ0, classTag[T0])), body, optimize)
  }

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    body: IR,
    optimize: Boolean): (PType, (Int, Region) => AsmFunction3[Region, T0, Boolean, R]) =
    apply(ctx, name0, typ0, body, optimize, None)

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    body: IR): (PType, (Int, Region) => AsmFunction3[Region, T0, Boolean, R]) =
    apply(ctx, name0, typ0, body, true)

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    name1: String,
    typ1: PType,
    body: IR,
    print: Option[PrintWriter]): (PType, (Int, Region) => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {
    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](ctx, print, FastSeq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body, optimize = true)
  }

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    name1: String,
    typ1: PType,
    body: IR,
    print: Option[PrintWriter],
    optimize: Boolean): (PType, (Int, Region) => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {
    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](ctx, print, FastSeq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body, optimize = optimize)
  }

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    name1: String,
    typ1: PType,
    body: IR): (PType, (Int, Region) => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) =
    apply(ctx, name0, typ0, name1, typ1, body, None)

  def apply[
  T0: TypeInfo : ClassTag,
  T1: TypeInfo : ClassTag,
  T2: TypeInfo : ClassTag,
  R: TypeInfo : ClassTag
  ](ctx: ExecuteContext,
    name0: String,
    typ0: PType,
    name1: String,
    typ1: PType,
    name2: String,
    typ2: PType,
    body: IR
  ): (PType, (Int, Region) => AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R]) = {
    apply[AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R], R](ctx, None, FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]),
      (name2, typ2, classTag[T2])
    ), body,
      optimize = true)
  }

  def apply[
  T0: TypeInfo : ClassTag,
  T1: TypeInfo : ClassTag,
  T2: TypeInfo : ClassTag,
  T3: TypeInfo : ClassTag,
  R: TypeInfo : ClassTag
  ](ctx: ExecuteContext,
    name0: String, typ0: PType,
    name1: String, typ1: PType,
    name2: String, typ2: PType,
    name3: String, typ3: PType,
    body: IR
  ): (PType, (Int, Region) => AsmFunction9[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, R]) = {
    apply[AsmFunction9[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, R], R](ctx, None, FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]),
      (name2, typ2, classTag[T2]),
      (name3, typ3, classTag[T3])
    ), body,
      optimize = true)
  }

  def apply[
  T0: ClassTag,
  T1: ClassTag,
  T2: ClassTag,
  T3: ClassTag,
  T4: ClassTag,
  T5: ClassTag,
  R: TypeInfo : ClassTag
  ](ctx: ExecuteContext,
    name0: String, typ0: PType,
    name1: String, typ1: PType,
    name2: String, typ2: PType,
    name3: String, typ3: PType,
    name4: String, typ4: PType,
    name5: String, typ5: PType,
    body: IR
  ): (PType, (Int, Region) => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]) = {

    apply[AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R], R](ctx, None, FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]),
      (name2, typ2, classTag[T2]),
      (name3, typ3, classTag[T3]),
      (name4, typ4, classTag[T4]),
      (name5, typ5, classTag[T5])
    ), body,
      optimize = true)
  }
}

object CompileWithAggregators2 {
  private[this] val codeCache: Cache[CodeCacheKey, CodeCacheValue] = new Cache(50)

  private def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStatePhysicalSignature],
    args: Seq[(String, PType, ClassTag[_])],
    argTypeInfo: Array[MaybeGenericTypeInfo[_]],
    body: IR,
    optimize: Boolean
  ): (PType, (Int, Region) => (F with FunctionWithAggRegion)) = {
    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(args.map { case (n, _, _) => n -> n }: _*))
    val k = CodeCacheKey(aggSigs, args.map { case (n, pt, _) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
      case None =>
    }

    val fb = EmitFunctionBuilder[F]("CompiledWithAggs", argTypeInfo, GenericTypeInfo[R]())

    var ir = body
    ir = Subst(ir, BindingEnv(args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer.apply(ctx, ir, optimize).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv(Env.fromSeq[Type](args.map { case (name, t, _) => name -> t.virtualType })))

    InferPType(ir, Env(args.map { case (n, pt, _) => n -> pt}: _*), aggSigs, null, null)

    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])

    Emit(ctx, ir, fb, Some(aggSigs))

    val f = fb.resultWithIndex()
    codeCache += k -> CodeCacheValue(ir.pType, f)
    (ir.pType, f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
  }

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStatePhysicalSignature],
    args: Seq[(String, PType, ClassTag[_])],
    body: IR,
    optimize: Boolean
  ): (PType, (Int, Region) => (F with FunctionWithAggRegion)) = {
    assert(args.forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t.virtualType) == ct })

    val ab = new ArrayBuilder[MaybeGenericTypeInfo[_]]()
    ab += GenericTypeInfo[Region]()
    args.foreach { case (_, t, _) =>
      ab += GenericTypeInfo()(typeToTypeInfo(t))
      ab += GenericTypeInfo[Boolean]()
    }

    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] = ab.result()

    CompileWithAggregators2[F, R](ctx, aggSigs, args, argTypeInfo, body, optimize)
  }

  def apply[R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStatePhysicalSignature],
    body: IR): (PType, (Int, Region) => AsmFunction1[Region, R] with FunctionWithAggRegion) = {

    apply[AsmFunction1[Region, R], R](ctx, aggSigs, FastSeq[(String, PType, ClassTag[_])](), body, optimize = true)
  }

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStatePhysicalSignature],
    name0: String, typ0: PType,
    body: IR): (PType, (Int, Region) => AsmFunction3[Region, T0, Boolean, R] with FunctionWithAggRegion) = {

    apply[AsmFunction3[Region, T0, Boolean, R], R](ctx, aggSigs, FastSeq((name0, typ0, classTag[T0])), body, optimize = true)
  }

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStatePhysicalSignature],
    name0: String, typ0: PType,
    name1: String, typ1: PType,
    body: IR): (PType, (Int, Region) => (AsmFunction5[Region, T0, Boolean, T1, Boolean, R] with FunctionWithAggRegion)) = {

    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](ctx, aggSigs, FastSeq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body, optimize = true)
  }
}


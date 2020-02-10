package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical.{PType, PBaseStruct}
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

    val fb = new EmitFunctionBuilder[F](argTypeInfo, GenericTypeInfo[R]())

    var ir = body
    ir = Subst(ir, BindingEnv(args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer.apply(ctx, ir, optimize).asInstanceOf[IR]
    TypeCheck(ir, BindingEnv.empty)
    InferPType(if (HasIRSharing(ir)) ir.deepCopy() else ir, Env(args.map { case (n, pt, _) => n -> pt}: _*))

    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])

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
    aggSigs: Array[AggStateSignature],
    args: Seq[(String, PType, ClassTag[_])],
    argTypeInfo: Array[MaybeGenericTypeInfo[_]],
    body: IR,
    optimize: Boolean
  ): (PType, (Int, Region) => (F with FunctionWithAggRegion)) = {
    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(args.map { case (n, _, _) => n -> n }: _*))
    val pAggSigs = aggSigs.map(_.toCanonicalPhysical)
    val k = CodeCacheKey(pAggSigs.toFastIndexedSeq, args.map { case (n, pt, _) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
      case None =>
    }

    val fb = new EmitFunctionBuilder[F](argTypeInfo, GenericTypeInfo[R]())

    var ir = body
    if (optimize)
      ir = Optimize(ir, noisy = true, context = "Compile", ctx)
    TypeCheck(ir, BindingEnv(Env.fromSeq[Type](args.map { case (name, t, _) => name -> t.virtualType })))

    val env = args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }

    ir = Subst(ir, BindingEnv(args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer.apply(ctx, ir, optimize).asInstanceOf[IR]
    TypeCheck(ir, BindingEnv.empty)
    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])

    Emit(ctx, ir, fb, Some(pAggSigs))

    val f = fb.resultWithIndex()
    codeCache += k -> CodeCacheValue(ir.pType, f)
    (ir.pType, f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
  }

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStateSignature],
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
    aggSigs: Array[AggStateSignature],
    body: IR): (PType, (Int, Region) => AsmFunction1[Region, R] with FunctionWithAggRegion) = {

    apply[AsmFunction1[Region, R], R](ctx, aggSigs, FastSeq[(String, PType, ClassTag[_])](), body, optimize = true)
  }

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStateSignature],
    name0: String, typ0: PType,
    body: IR): (PType, (Int, Region) => AsmFunction3[Region, T0, Boolean, R] with FunctionWithAggRegion) = {

    apply[AsmFunction3[Region, T0, Boolean, R], R](ctx, aggSigs, FastSeq((name0, typ0, classTag[T0])), body, optimize = true)
  }

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    ctx: ExecuteContext,
    aggSigs: Array[AggStateSignature],
    name0: String, typ0: PType,
    name1: String, typ1: PType,
    body: IR): (PType, (Int, Region) => (AsmFunction5[Region, T0, Boolean, T1, Boolean, R] with FunctionWithAggRegion)) = {

    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](ctx, aggSigs, FastSeq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body, optimize = true)
  }
}

object CompileIterator {
  import is.hail.asm4s.joinpoint._

  private trait RegionValueIteratorWrapper extends Iterator[RegionValue] {
    def step(): Boolean

    protected def setRV(r: Region, offset: Long) {
      rv.set(r, offset)
    }
    private val rv = RegionValue()
    private var _stepped = false
    private var _hasNext = false
    def hasNext: Boolean = {
      if (!_stepped) {
        _hasNext = step()
        _stepped = true
      }
      _hasNext
    }
    def next(): RegionValue = {
      if (!hasNext)
        return Iterator.empty.next()
      _stepped = false
      rv
    }
  }

  private def compileStepper[F >: Null: TypeInfo](
    ctx: ExecuteContext,
    ir: IR,
    argTypeInfo: Array[MaybeGenericTypeInfo[_]],
    printWriter: Option[PrintWriter]
  ): (PType, (Int, Region) => F) = {

    val fb = new EmitFunctionBuilder[F](argTypeInfo, GenericTypeInfo[Boolean], namePrefix = "stream")
    val stepF = fb.apply_method

    val self = Code.checkcast[RegionValueIteratorWrapper](stepF.getArg[Object](0).load)
    val er = EmitRegion.default(stepF)
    val emitter = new Emit(ctx, stepF)

    val EmitStream(stream, eltPType) = EmitStream(emitter, ir, Env.empty, er, None)
    implicit val statePP = stream.stateP
    val state = statePP.newFields(fb, "state")
    assert(eltPType.isInstanceOf[PBaseStruct])

    val didInit = fb.newField[Boolean]("did_init")
    fb.addInitInstructions(didInit := false)

    stepF.emit(JoinPoint.CallCC[Code[Boolean]] { (jb, ret) =>
      val step = jb.joinPoint()
      step.define(_ => stream.step(stepF, jb, state.load) {
        case EmitStream.EOS => ret(false)
        case EmitStream.Yield(elt, s1) =>
          Code(
            elt.setup,
            elt.m.mux(Code._fatal("empty row!"),
                      self.invoke[Region, Long, Unit]("setRV", er.region, elt.value)),
            state := s1,
            ret(true))
      })
      didInit.mux(step(()), stream.init(stepF, jb, ()) {
        case EmitStream.Missing => Code._fatal("missing stream!")
        case EmitStream.Start(s0) => Code(didInit := true, state := s0, step(()))
      })
    })

    (eltPType, fb.resultWithIndex(printWriter))
  }

  def apply(
    ctx: ExecuteContext,
    ir: IR
  ): (PType, (Int, Region) => Iterator[RegionValue]) = {
    val (eltPType, makeStepper) = compileStepper[AsmFunction1[Region, Boolean]](
      ctx, ir,
      Array[MaybeGenericTypeInfo[_]](
        GenericTypeInfo[Region], GenericTypeInfo[RegionValue]),
      None)
    (eltPType, (idx, r) => {
      val stepper = makeStepper(idx, r)
      new RegionValueIteratorWrapper {
        def step(): Boolean = stepper(r)
      }
    })
  }

  def apply[T0: TypeInfo](
    ctx: ExecuteContext,
    typ0: PType,
    ir: IR
  ): (PType, (Int, Region, T0, Boolean) => Iterator[RegionValue]) = {
    val (eltPType, makeStepper) = compileStepper[AsmFunction3[Region, T0, Boolean, Boolean]](
      ctx, ir,
      Array[MaybeGenericTypeInfo[_]](
        GenericTypeInfo[Region], GenericTypeInfo[RegionValue],
        GenericTypeInfo[T0], GenericTypeInfo[Boolean]),
      None)
    (eltPType, (idx, r, v0, m0) => {
      val stepper = makeStepper(idx, r)
      new RegionValueIteratorWrapper {
        def step(): Boolean = stepper(r, v0, m0)
      }
    })
  }

  def apply[T0: TypeInfo, T1: TypeInfo](
    ctx: ExecuteContext,
    typ0: PType, typ1: PType,
    ir: IR
  ): (PType, (Int, Region, T0, Boolean, T1, Boolean) => Iterator[RegionValue]) = {
    val (eltPType, makeStepper) = compileStepper[AsmFunction5[Region, T0, Boolean, T1, Boolean, Boolean]](
      ctx, ir,
      Array[MaybeGenericTypeInfo[_]](
        GenericTypeInfo[Region], GenericTypeInfo[RegionValue],
        GenericTypeInfo[T0], GenericTypeInfo[Boolean],
        GenericTypeInfo[T1], GenericTypeInfo[Boolean]),
      None)
    (eltPType, (idx, r, v0, m0, v1, m1) => {
      val stepper = makeStepper(idx, r)
      new RegionValueIteratorWrapper {
        def step(): Boolean = stepper(r, v0, m0, v1, m1)
      }
    })
  }
}

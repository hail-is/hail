package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}

object Compile {

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    args: Seq[(String, Type, ClassTag[_])],
    body: IR,
    nSpecialArgs: Int
  ): (Type, () => F) = {
    assert(args.forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    val ab = new ArrayBuilder[MaybeGenericTypeInfo[_]]()
    ab += GenericTypeInfo[Region]()
    if (nSpecialArgs == 2)
      ab += GenericTypeInfo[Array[RegionValueAggregator]]()
    args.foreach { case (_, t, _) =>
      ab += GenericTypeInfo()(typeToTypeInfo(t))
      ab += GenericTypeInfo[Boolean]()
    }

    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] = ab.result()

    Compile[F, R](args, argTypeInfo, body, nSpecialArgs)
  }

  private def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    args: Seq[(String, Type, ClassTag[_])],
    argTypeInfo: Array[MaybeGenericTypeInfo[_]],
    body: IR,
    nSpecialArgs: Int
  ): (Type, () => F) = {
    val fb = new EmitFunctionBuilder[F](argTypeInfo, GenericTypeInfo[R]())

    var ir = body
    val env = args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }

    ir = Subst(ir, env)
    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])
    Emit(ir, fb, nSpecialArgs)
    (ir.typ, fb.result())
  }

  def apply[R: TypeInfo : ClassTag](body: IR): (Type, () => AsmFunction1[Region, R]) = {
    apply[AsmFunction1[Region, R], R](FastSeq[(String, Type, ClassTag[_])](), body, 1)
  }

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    body: IR): (Type, () => AsmFunction3[Region, T0, Boolean, R]) = {

    apply[AsmFunction3[Region, T0, Boolean, R], R](FastSeq((name0, typ0, classTag[T0])), body, 1)
  }

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    body: IR): (Type, () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {

    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](FastSeq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body, 1)
  }

  def apply[
  T0: TypeInfo : ClassTag,
  T1: TypeInfo : ClassTag,
  T2: TypeInfo : ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    name2: String,
    typ2: Type,
    body: IR
  ): (Type, () => AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R]) = {
    apply[AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R], R](FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]),
      (name2, typ2, classTag[T2])
    ), body, 1)
  }

  def apply[
  T0: ClassTag,
  T1: ClassTag,
  T2: ClassTag,
  T3: ClassTag,
  T4: ClassTag,
  T5: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    name1: String, typ1: Type,
    name2: String, typ2: Type,
    name3: String, typ3: Type,
    name4: String, typ4: Type,
    name5: String, typ5: Type,
    body: IR
  ): (Type, () => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]) = {

    apply[AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R], R](FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]),
      (name2, typ2, classTag[T2]),
      (name3, typ3, classTag[T3]),
      (name4, typ4, classTag[T4]),
      (name5, typ5, classTag[T5])
    ), body, 1)
  }
}

object CompileWithAggregators {
  type Compiler[F] = (IR) => (Type, () => F)
  type IRAggFun1[T0] =
    AsmFunction4[Region, Array[RegionValueAggregator], T0, Boolean, Unit]
  type IRAggFun2[T0, T1] =
    AsmFunction6[Region, Array[RegionValueAggregator],
      T0, Boolean,
      T1, Boolean,
      Unit]
  type IRAggFun3[T0, T1, T2] =
    AsmFunction8[Region, Array[RegionValueAggregator],
      T0, Boolean,
      T1, Boolean,
      T2, Boolean,
      Unit]
  type IRAggFun4[T0, T1, T2, T3] =
    AsmFunction10[Region, Array[RegionValueAggregator],
      T0, Boolean,
      T1, Boolean,
      T2, Boolean,
      T3, Boolean,
      Unit]
  type IRFun1[T0, R] =
    AsmFunction3[Region, T0, Boolean, R]
  type IRFun2[T0, T1, R] =
    AsmFunction5[Region, T0, Boolean, T1, Boolean, R]
  type IRFun3[T0, T1, T2, R] =
    AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R]

  def toIRs[
  F0 >: Null : TypeInfo,
  F1 >: Null : TypeInfo,
  F2 >: Null : TypeInfo,
  R: TypeInfo : ClassTag
  ](args: Seq[(String, Type, ClassTag[_])],
    aggScopeArgs: Seq[(String, Type, ClassTag[_])],
    body: IR
  ): (Array[RegionValueAggregator], (IR, Compiler[F0]), (IR, Compiler[F1]), Type, (IR, Compiler[F2])) = {
    assert((args ++ aggScopeArgs).forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    val (postAggIR, aggResultType, initOpIR, seqOpIR, rvAggs) = ExtractAggregators(body)
    val compileInitOp = (initOp: IR) => Compile[F0, Unit](args, initOp, 2)
    val compileSeqOp = (seqOp: IR) => Compile[F1, Unit](aggScopeArgs, seqOp, 2)

    val args2 = ("AGGR", aggResultType, classTag[Long]) +: args
    val compilePostAgg = (postAgg: IR) => Compile[F2, R](args2, postAgg, 1)
    (rvAggs,
      (initOpIR, compileInitOp),
      (seqOpIR, compileSeqOp),
      aggResultType,
      (postAggIR, compilePostAgg))
  }

  def apply[
  F0 >: Null : TypeInfo,
  F1 >: Null : TypeInfo,
  F2 >: Null : TypeInfo,
  R: TypeInfo : ClassTag
  ](args: Seq[(String, Type, ClassTag[_])],
    aggScopeArgs: Seq[(String, Type, ClassTag[_])],
    body: IR,
    transformInitOp: (Int, IR) => IR,
    transformSeqOp: (Int, IR) => IR
  ): (Array[RegionValueAggregator], () => F0, () => F1, Type, () => F2, Type) = {
    val (
      rvAggs,
      (initOpIR, compileInitOp),
      (seqOpIR, compileSeqOp),
      aggResultType,
      (postAggIR, compilePostAgg)
    ) = toIRs[F0, F1, F2, R](args, aggScopeArgs, body)
    val nAggs = rvAggs.length
    val (_, initOps) = compileInitOp(trace("initop", transformInitOp(nAggs, initOpIR)))
    val (_, seqOps) = compileSeqOp(trace("seqop", transformSeqOp(nAggs, seqOpIR)))
    val (t, f) = compilePostAgg(postAggIR)
    (rvAggs, initOps, seqOps, aggResultType, f, t)
  }

  private[this] def trace(name: String, t: IR): IR = {
    log.info(name + " " + Pretty(t))
    t
  }

  def apply[
  T0: ClassTag,
  S0: ClassTag,
  S1: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    aggName0: String, aggTyp0: Type,
    aggName1: String, aggTyp1: Type,
    body: IR,
    transformInitOp: (Int, IR) => IR,
    transformSeqOp: (Int, IR) => IR
  ): (Array[RegionValueAggregator],
    () => IRAggFun1[T0],
    () => IRAggFun2[S0, S1],
    Type,
    () => IRFun2[Long, T0, R],
    Type) = {
    val args = FastSeq((name0, typ0, classTag[T0]))

    val aggScopeArgs = FastSeq(
      (aggName0, aggTyp0, classTag[S0]),
      (aggName1, aggTyp1, classTag[S1]))

    apply[IRAggFun1[T0], IRAggFun2[S0, S1], IRFun2[Long, T0, R], R
    ](args, aggScopeArgs, body, transformInitOp, transformSeqOp)
  }

  def apply[
  T0: ClassTag,
  T1: ClassTag,
  S0: ClassTag,
  S1: ClassTag,
  S2: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    name1: String, typ1: Type,
    aggName0: String, aggType0: Type,
    aggName1: String, aggType1: Type,
    aggName2: String, aggType2: Type,
    body: IR,
    transformInitOp: (Int, IR) => IR,
    transformSeqOp: (Int, IR) => IR
  ): (Array[RegionValueAggregator],
    () => IRAggFun2[T0, T1],
    () => IRAggFun3[S0, S1, S2],
    Type,
    () => IRFun3[Long, T0, T1, R],
    Type) = {
    val args = FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]))

    val aggArgs = FastSeq(
      (aggName0, aggType0, classTag[S0]),
      (aggName1, aggType1, classTag[S1]),
      (aggName2, aggType2, classTag[S2]))

    apply[IRAggFun2[T0, T1], IRAggFun3[S0, S1, S2], IRFun3[Long, T0, T1, R], R
    ](args, aggArgs, body, transformInitOp, transformSeqOp)
  }

  def toIRs[
  T0: ClassTag,
  T1: ClassTag,
  S0: ClassTag,
  S1: ClassTag,
  S2: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    name1: String, typ1: Type,
    aggName0: String, aggType0: Type,
    aggName1: String, aggType1: Type,
    aggName2: String, aggType2: Type,
    body: IR
  ): (Array[RegionValueAggregator],
    (IR, Compiler[IRAggFun2[T0, T1]]),
    (IR, Compiler[IRAggFun3[S0, S1, S2]]),
    Type,
    (IR, Compiler[IRFun3[Long, T0, T1, R]])) = {
    val args = FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]))

    val aggArgs = FastSeq(
      (aggName0, aggType0, classTag[S0]),
      (aggName1, aggType1, classTag[S1]),
      (aggName2, aggType2, classTag[S2]))

    toIRs[IRAggFun2[T0, T1], IRAggFun3[S0, S1, S2], IRFun3[Long, T0, T1, R], R
    ](args, aggArgs, body)
  }

  def apply[
    T0: ClassTag,
    S0: ClassTag,
    S1: ClassTag,
    S2: ClassTag,
    R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    aggName0: String, aggType0: Type,
    aggName1: String, aggType1: Type,
    aggName2: String, aggType2: Type,
    body: IR,
    transformInitOp: (Int, IR) => IR,
    transformAggIR: (Int, IR) => IR
  ): (Array[RegionValueAggregator],
    () => IRAggFun1[T0],
    () => IRAggFun3[S0, S1, S2],
    Type,
    () => IRFun1[T0, R],
    Type) = {
    val args = FastSeq(
      (name0, typ0, classTag[T0]))

    val aggArgs = FastSeq(
      (aggName0, aggType0, classTag[S0]),
      (aggName1, aggType1, classTag[S1]),
      (aggName2, aggType2, classTag[S2]))

    apply[IRAggFun1[T0], IRAggFun3[S0, S1, S2], IRFun1[T0, R], R
    ](args, aggArgs, body, transformInitOp, transformAggIR)
  }

  def apply[
  T0: ClassTag,
  T1: ClassTag,
  T2: ClassTag,
  S0: ClassTag,
  S1: ClassTag,
  S2: ClassTag,
  S3: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    name1: String, typ1: Type,
    name2: String, typ2: Type,
    aggName0: String, aggType0: Type,
    aggName1: String, aggType1: Type,
    aggName2: String, aggType2: Type,
    aggName3: String, aggType3: Type,
    body: IR,
    transformInitOp: (Int, IR) => IR,
    transformSeqOp: (Int, IR) => IR
  ): (Array[RegionValueAggregator],
    () => IRAggFun3[T0, T1, T2],
    () => IRAggFun4[S0, S1, S2, S3],
    Type,
    () => IRFun3[T0, T1, T2, R],
    Type) = {
    val args = FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]),
      (name2, typ2, classTag[T2]))

    val aggArgs = FastSeq(
      (aggName0, aggType0, classTag[S0]),
      (aggName1, aggType1, classTag[S1]),
      (aggName2, aggType2, classTag[S2]),
      (aggName3, aggType3, classTag[S3]))

    apply[IRAggFun3[T0, T1, T2], IRAggFun4[S0, S1, S2, S3], IRFun3[T0, T1, T2, R], R
    ](args, aggArgs, body, transformInitOp, transformSeqOp)
  }
}

package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils.FastSeq

import scala.reflect.{ClassTag, classTag}

object Compile {

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](args: Seq[(String, Type, ClassTag[_])], body: IR): (Type, () => F) = {
    assert(args.forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] =
      GenericTypeInfo[Region]() +:
        args.flatMap { case (_, t, _) =>
          FastSeq[GenericTypeInfo[_]](GenericTypeInfo()(typeToTypeInfo(t)), GenericTypeInfo[Boolean]())
        }.toArray

    Compile[F, R](args, argTypeInfo, body, None)
  }

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    args: Seq[(String, Type, ClassTag[_])],
    body: IR,
    tAggIn: TAggregable
  ): (Type, () => F) = {
    assert(args.forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] =
      GenericTypeInfo[Region]() +:
        GenericTypeInfo[Array[RegionValueAggregator]]() +:
        args.flatMap { case (_, t, _) =>
          FastSeq[GenericTypeInfo[_]](GenericTypeInfo()(typeToTypeInfo(t)), GenericTypeInfo[Boolean]())
        }.toArray

    Compile[F, R](args, argTypeInfo, body, Some(tAggIn))
  }

  private def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](
    args: Seq[(String, Type, ClassTag[_])],
    argTypeInfo: Array[MaybeGenericTypeInfo[_]],
    body: IR,
    tAggIn: Option[TAggregable]
  ): (Type, () => F) = {
    val fb = new EmitFunctionBuilder[F](argTypeInfo, GenericTypeInfo[R]())

    var ir = body
    val env = args
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }

    ir = Subst(ir, env)
    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])
    Emit(ir, fb, tAggIn, if (tAggIn.isDefined) 2 else 1)
    (ir.typ, fb.result())

  }

  def apply[R: TypeInfo : ClassTag](body: IR): (Type, () => AsmFunction1[Region, R]) = {
    apply[AsmFunction1[Region, R], R](FastSeq[(String, Type, ClassTag[_])](), body)
  }

  def apply[T0: ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    body: IR): (Type, () => AsmFunction3[Region, T0, Boolean, R]) = {

    apply[AsmFunction3[Region, T0, Boolean, R], R](FastSeq((name0, typ0, classTag[T0])), body)
  }

  def apply[T0: ClassTag, T1: ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    body: IR): (Type, () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {

    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](FastSeq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body)
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
    ), body)
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
    ), body)
  }
}

object CompileWithAggregators {
  def apply[
  F1 >: Null : TypeInfo,
  F2 >: Null : TypeInfo,
  R: TypeInfo : ClassTag
  ](aggName: String,
    aggType: TAggregable,
    args: Seq[(String, Type, ClassTag[_])],
    aggScopeArgs: Seq[(String, Type, ClassTag[_])],
    body: IR,
    transformAggIR: (Int, IR) => IR
  ): (Array[RegionValueAggregator], () => F1, Type, () => F2, Type) = {

    assert((args ++ aggScopeArgs).forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    val env = ((aggName, aggType, TypeToIRIntermediateClassTag(aggType)) +: args).zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }

    val ir = Subst(body, env)

    val (postAggIR, aggResultType, aggIR, rvAggs) = ExtractAggregators(ir, aggType)
    val nAggs = rvAggs.length
    val (_, seqOps) = Compile[F1, Unit](aggScopeArgs, transformAggIR(nAggs, aggIR), aggType)

    val args2 = ("AGGR", aggResultType, classTag[Long]) +: args
    val (t, f) = Compile[F2, R](args2, postAggIR)
    (rvAggs, seqOps, aggResultType, f, t)
  }

  def apply[
  T0: ClassTag,
  TAGG: ClassTag,
  S0: ClassTag,
  S1: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    aggName: String, aggTyp: TAggregable,
    aggName0: String, aggTyp0: Type,
    aggName1: String, aggTyp1: Type,
    body: IR
  ): (Array[RegionValueAggregator],
    () => AsmFunction8[Region, Array[RegionValueAggregator], TAGG, Boolean, S0, Boolean, S1, Boolean, Unit],
    Type,
    () => AsmFunction5[Region, Long, Boolean, T0, Boolean, R],
    Type) = {

    assert(TypeToIRIntermediateClassTag(aggTyp) == classTag[TAGG])

    val args = FastSeq((name0, typ0, classTag[T0]))

    val aggScopeArgs = FastSeq(
      (aggName, aggTyp, classTag[TAGG]),
      (aggName0, aggTyp0, classTag[S0]),
      (aggName1, aggTyp1, classTag[S1]))

    apply[AsmFunction8[Region, Array[RegionValueAggregator], TAGG, Boolean, S0, Boolean, S1, Boolean, Unit],
      AsmFunction5[Region, Long, Boolean, T0, Boolean, R],
      R](aggName, aggTyp, args, aggScopeArgs, body, (nAggs: Int, aggIR: IR) => aggIR)
  }

  def apply[
  T0: ClassTag,
  T1: ClassTag,
  TAGG: ClassTag,
  S0: ClassTag,
  S1: ClassTag,
  S2: ClassTag,
  R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    name1: String, typ1: Type,
    aggName: String, aggTyp: TAggregable,
    aggName0: String, aggType0: Type,
    aggName1: String, aggType1: Type,
    aggName2: String, aggType2: Type,
    body: IR,
    transformAggIR: (Int, IR) => IR
  ): (Array[RegionValueAggregator],
    () => AsmFunction10[Region, Array[RegionValueAggregator], TAGG, Boolean, S0, Boolean, S1, Boolean, S2, Boolean, Unit],
    Type,
    () => AsmFunction7[Region, Long, Boolean, T0, Boolean, T1, Boolean, R],
    Type) = {

    assert(TypeToIRIntermediateClassTag(aggTyp) == classTag[TAGG])

    val args = FastSeq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]))

    val aggArgs = FastSeq(
      (aggName, aggTyp.elementType, classTag[TAGG]),
      (aggName0, aggType0, classTag[S0]),
      (aggName1, aggType1, classTag[S1]),
      (aggName2, aggType2, classTag[S2]))

    apply[AsmFunction10[Region, Array[RegionValueAggregator], TAGG, Boolean, S0, Boolean, S1, Boolean, S2, Boolean, Unit],
      AsmFunction7[Region, Long, Boolean, T0, Boolean, T1, Boolean, R],
      R](aggName, aggTyp, args, aggArgs, body, transformAggIR)
  }
}

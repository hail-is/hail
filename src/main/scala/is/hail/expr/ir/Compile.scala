package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.asm4s._
import is.hail.expr.types._

import scala.reflect.{ClassTag, classTag}

object Compile {

  def apply[F >: Null : TypeInfo, R: TypeInfo : ClassTag](args: Seq[(String, Type, ClassTag[_])], body: IR): (Type, () => F) = {
    assert(args.forall{ case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    val argTypeInfo: Array[GenericTypeInfo[_]] =
      GenericTypeInfo[Region]() +:
        args.flatMap { case (_, t, _) =>
          List[GenericTypeInfo[_]](GenericTypeInfo()(typeToTypeInfo(t)), GenericTypeInfo[Boolean]()).iterator
        }.toArray

    val fb = new FunctionBuilder[F](argTypeInfo.asInstanceOf[Array[MaybeGenericTypeInfo[_]]], GenericTypeInfo[R]())

    var ir = body
    val env = args
      .zipWithIndex
      .foldLeft(new Env[IR]()) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }

    ir = Subst(ir, env)
    Infer(ir)
    assert(TypeToIRIntermediateClassTag(ir.typ) == classTag[R])
    Emit(ir, fb)
    (ir.typ, fb.result())
  }

  def apply[R: TypeInfo : ClassTag](body: IR): (Type, () => AsmFunction1[Region, R]) = {
    apply[AsmFunction1[Region, R], R](Seq(), body)
  }

  def apply[T0 : ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    body: IR): (Type, () => AsmFunction3[Region, T0, Boolean, R]) = {

    apply[AsmFunction3[Region, T0, Boolean, R], R](Seq((name0, typ0, classTag[T0])), body)
  }

  def apply[T0 : ClassTag, T1 : ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    body: IR): (Type, () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {

    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](Seq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body)
  }

  def apply[
    T0 : TypeInfo : ClassTag,
    T1 : TypeInfo : ClassTag,
    T2 : TypeInfo : ClassTag,
    R: TypeInfo : ClassTag
  ](name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    name2: String,
    typ2: Type,
    body: IR
  ): (Type, () => AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R]) = {
    assert(TypeToIRIntermediateClassTag(typ0) == classTag[T0])
    assert(TypeToIRIntermediateClassTag(typ1) == classTag[T1])
    assert(TypeToIRIntermediateClassTag(typ2) == classTag[T2])
    val fb = FunctionBuilder.functionBuilder[Region, T0, Boolean, T1, Boolean, T2, Boolean, R]
    var e = body
    val env = new Env[IR]()
      .bind(name0, In(0, typ0))
      .bind(name1, In(1, typ1))
      .bind(name2, In(2, typ2))
    e = Subst(e, env)
    Infer(e)
    assert(TypeToIRIntermediateClassTag(e.typ) == classTag[R])
    Emit(e, fb)
    (e.typ, fb.result())
  }

  def apply[
    T0 : ClassTag,
    T1 : ClassTag,
    T2 : ClassTag,
    T3 : ClassTag,
    T4 : ClassTag,
    T5 : ClassTag,
    R: TypeInfo : ClassTag
  ](name0: String, typ0: Type,
    name1: String, typ1: Type,
    name2: String, typ2: Type,
    name3: String, typ3: Type,
    name4: String, typ4: Type,
    name5: String, typ5: Type,
    body: IR
  ): (Type, () => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]) = {

    apply[AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R], R](Seq(
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
    body: IR
  ): (Array[RegionValueAggregator], Array[() => F1], Type, () => F2, Type) = {

    assert((args ++ aggScopeArgs).forall { case (_, t, ct) => TypeToIRIntermediateClassTag(t) == ct })

    // the aggregable must be argument 0 to post-aggregated IR, so must have a place holder in env
    val env = ((aggName, aggType, TypeToIRIntermediateClassTag(aggType)) +: args)
      .zipWithIndex
      .foldLeft(new Env[IR]()) { case (e, ((n, t, _), i)) => e.bind(n, In(i, t)) }

    val ir = Subst(body, env)
    Infer(ir, Some(aggType))

    val (postAggIR, aggResultType, aggregators) = ExtractAggregators(ir, aggType)

    val f1TypeInfo: Array[GenericTypeInfo[_]] =
      GenericTypeInfo[Region]() +:
        GenericTypeInfo[RegionValueAggregator]() +:
        (aggType +: aggScopeArgs.map(_._2).toArray).flatMap { t =>
          List[GenericTypeInfo[_]](GenericTypeInfo()(typeToTypeInfo(t)), GenericTypeInfo[Boolean]()).iterator
        }

    val (rvAggs, seqOps) = aggregators.zipWithIndex.map { case ((ir, agg), i) =>
      val fb = new FunctionBuilder[F1](f1TypeInfo.asInstanceOf[Array[MaybeGenericTypeInfo[_]]], GenericTypeInfo[Unit]())
      Emit(ir, fb, 2, aggType)
      (agg, fb.result())
    }.unzip

    val args1 = ("AGGR", aggResultType, classTag[Long]) +: args
    val (t, f) = Compile[F2, R](args1, postAggIR)
    (rvAggs, seqOps, aggResultType, f, t)
  }

  def apply[
    TAGG : ClassTag,
    T0 : ClassTag,
    S0 : ClassTag,
    S1 : ClassTag,
    R: TypeInfo : ClassTag
  ](aggName: String,
    aggTyp: TAggregable,
    name0: String,
    typ0: Type,
    body: IR
  ): (Array[RegionValueAggregator],
    Array[() => AsmFunction8[Region, RegionValueAggregator, TAGG, Boolean, S0, Boolean, S1, Boolean, Unit]],
    Type,
    () => AsmFunction5[Region, Long, Boolean, T0, Boolean, R],
    Type) = {

    assert(TypeToIRIntermediateClassTag(aggTyp) == classTag[TAGG])

    val args = Seq((name0, typ0, classTag[T0]))

    val scope = aggTyp.symTab
    assert(scope.size == 2)

    val aggScopeArgs = scope
      .map { case (n, (i, t)) => (n, t) }
      .zip(Array(classTag[S0], classTag[S1]))
      .map { case ((n, t), ct) => (n, t, ct) }.toArray

    apply[AsmFunction8[Region, RegionValueAggregator, TAGG, Boolean, S0, Boolean, S1, Boolean, Unit],
      AsmFunction5[Region, Long, Boolean, T0, Boolean, R],
      R](aggName, aggTyp, args, aggScopeArgs, body)
  }

  def apply[
    TAGG : ClassTag,
    T0 : ClassTag,
    T1 : ClassTag,
    S0 : ClassTag,
    S1 : ClassTag,
    S2 : ClassTag,
    S3 : ClassTag,
    R: TypeInfo : ClassTag
  ](aggName: String,
    aggTyp: TAggregable,
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    body: IR
  ): (Array[RegionValueAggregator],
    Array[() => AsmFunction12[Region, RegionValueAggregator, TAGG, Boolean, S0, Boolean, S1, Boolean, S2, Boolean, S3, Boolean, Unit]],
    Type,
    () => AsmFunction7[Region, Long, Boolean, T0, Boolean, T1, Boolean, R],
    Type) = {

    assert(TypeToIRIntermediateClassTag(aggTyp) == classTag[TAGG])

    val args = Seq(
      (name0, typ0, classTag[T0]),
      (name1, typ1, classTag[T1]))

    val scope = aggTyp.symTab
    assert(scope.size == 4)

    val aggScopeArgs = scope
      .map { case (n, (i, t)) => (n, t) }
      .zip(Array(classTag[S0], classTag[S1], classTag[S2], classTag[S3]))
      .map { case ((n, t), ct) => (n, t, ct) }.toArray

    apply[AsmFunction12[Region, RegionValueAggregator, TAGG, Boolean, S0, Boolean, S1, Boolean, S2, Boolean, S3, Boolean, Unit],
      AsmFunction7[Region, Long, Boolean, T0, Boolean, T1, Boolean, R],
      R](aggName, aggTyp, args, aggScopeArgs, body)
  }
}

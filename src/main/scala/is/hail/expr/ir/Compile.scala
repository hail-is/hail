package is.hail.expr.ir

import is.hail.annotations._
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

  def apply[T0: TypeInfo : ClassTag, T1: TypeInfo : ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    body: IR): (Type, () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {

    apply[AsmFunction5[Region, T0, Boolean, T1, Boolean, R], R](Seq((name0, typ0, classTag[T0]), (name1, typ1, classTag[T1])), body)
  }

  def apply[T0: TypeInfo : ClassTag, T1: TypeInfo : ClassTag, T2: TypeInfo : ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    name2: String,
    typ2: Type,
    body: IR): (Type, () => AsmFunction7[Region, T0, Boolean, T1, Boolean, T2, Boolean, R]) = {
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

  def apply[T0: TypeInfo : ClassTag, T1: TypeInfo : ClassTag, T2: TypeInfo : ClassTag,
  T3: TypeInfo : ClassTag, T4: TypeInfo : ClassTag, T5: TypeInfo : ClassTag,
  R: TypeInfo : ClassTag](
    name0: String, typ0: Type,
    name1: String, typ1: Type,
    name2: String, typ2: Type,
    name3: String, typ3: Type,
    name4: String, typ4: Type,
    name5: String, typ5: Type,
    body: IR): (Type, () => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]) = {

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

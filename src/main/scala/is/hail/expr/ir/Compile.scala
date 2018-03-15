package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types._

import scala.reflect.{ClassTag, classTag}

object Compile {

  def apply[R: TypeInfo : ClassTag](body: IR): (Type, () => AsmFunction1[Region, R]) = {
    val fb = FunctionBuilder.functionBuilder[Region, R]
    var e = body
    val env = new Env[IR]()
    e = Subst(e, env)
    Infer(e)
    assert(TypeToIRIntermediateClassTag(e.typ) == classTag[R])
    Emit(e, fb)
    (e.typ, fb.result())
  }

  def apply[T0: TypeInfo : ClassTag, T1: TypeInfo : ClassTag, R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    body: IR): (Type, () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R]) = {
    assert(TypeToIRIntermediateClassTag(typ0) == classTag[T0])
    assert(TypeToIRIntermediateClassTag(typ1) == classTag[T1])
    val fb = FunctionBuilder.functionBuilder[Region, T0, Boolean, T1, Boolean, R]
    var e = body
    val env = new Env[IR]()
      .bind(name0, In(0, typ0))
      .bind(name1, In(1, typ1))
    e = Subst(e, env)
    Infer(e)
    assert(TypeToIRIntermediateClassTag(e.typ) == classTag[R])
    Emit(e, fb)
    (e.typ, fb.result())
  }

  def apply[T0: TypeInfo : ClassTag, T1: TypeInfo : ClassTag, T2: TypeInfo : ClassTag,
  T3: TypeInfo : ClassTag, T4: TypeInfo : ClassTag, T5: TypeInfo : ClassTag,
  R: TypeInfo : ClassTag](
    name0: String,
    typ0: Type,
    name1: String,
    typ1: Type,
    name2: String,
    typ2: Type,
    name3: String,
    typ3: Type,
    name4: String,
    typ4: Type,
    name5: String,
    typ5: Type,
    body: IR): (Type, () => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]) = {
    assert(TypeToIRIntermediateClassTag(typ0) == classTag[T0])
    assert(TypeToIRIntermediateClassTag(typ1) == classTag[T1])
    assert(TypeToIRIntermediateClassTag(typ2) == classTag[T2])
    assert(TypeToIRIntermediateClassTag(typ3) == classTag[T3])
    assert(TypeToIRIntermediateClassTag(typ4) == classTag[T4])
    assert(TypeToIRIntermediateClassTag(typ5) == classTag[T5])
    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] =
      Array(GenericTypeInfo[Region],
        GenericTypeInfo[T0], GenericTypeInfo[Boolean],
        GenericTypeInfo[T1], GenericTypeInfo[Boolean],
        GenericTypeInfo[T2], GenericTypeInfo[Boolean],
        GenericTypeInfo[T3], GenericTypeInfo[Boolean],
        GenericTypeInfo[T4], GenericTypeInfo[Boolean],
        GenericTypeInfo[T5], GenericTypeInfo[Boolean])

    val args = Array((name0, typ0), (name1, typ1), (name2, typ2), (name3, typ3), (name4, typ4), (name5, typ5))

    val fb = new FunctionBuilder[AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]](argTypeInfo, GenericTypeInfo[R])
    var e = body
    val env = args.zipWithIndex.foldLeft(new Env[IR]()) {
      case (newEnv, ((name, rvr), i)) => newEnv.bind(name, In(i, rvr))
    }
    e = Subst(e, env)
    Infer(e)
    assert(TypeToIRIntermediateClassTag(e.typ) == classTag[R])
    Emit(e, fb)
    (e.typ, fb.result())
  }
}

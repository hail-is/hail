package is.hail.expr.ir

import java.io.FileOutputStream

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}

case class RegionValueRep[RVT: ClassTag](typ: Type) {
  assert(TypeToIRIntermediateClassTag(typ) == classTag[RVT])
}

object Compile {

  var i: Int = 0

  def apply[T0: TypeInfo, T1: TypeInfo, R: TypeInfo](
    name0: String,
    rep0: RegionValueRep[T0],
    name1: String,
    rep1: RegionValueRep[T1],
    rRep: RegionValueRep[R],
    body: IR): () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R] = {
    val fb = FunctionBuilder.functionBuilder[Region, T0, Boolean, T1, Boolean, R]
    var e = body
    val env = new Env[IR]()
      .bind(name0, In(0, rep0.typ))
      .bind(name1, In(1, rep1.typ))
    e = Subst(e, env)
    Infer(e)
    assert(e.typ == rRep.typ)
    Emit(e, fb)
    fb.result()
  }

  def apply[T0: TypeInfo, T1: TypeInfo, T2: TypeInfo, T3: TypeInfo, T4: TypeInfo, T5: TypeInfo, R: TypeInfo](
    name0: String,
    rep0: RegionValueRep[T0],
    name1: String,
    rep1: RegionValueRep[T1],
    name2: String,
    rep2: RegionValueRep[T2],
    name3: String,
    rep3: RegionValueRep[T3],
    name4: String,
    rep4: RegionValueRep[T4],
    name5: String,
    rep5: RegionValueRep[T5],
    rRep: RegionValueRep[R],
    body: IR): () => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R] = {
    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] =
      Array(GenericTypeInfo[Region],
        GenericTypeInfo[T0], GenericTypeInfo[Boolean],
        GenericTypeInfo[T1], GenericTypeInfo[Boolean],
        GenericTypeInfo[T2], GenericTypeInfo[Boolean],
        GenericTypeInfo[T3], GenericTypeInfo[Boolean],
        GenericTypeInfo[T4], GenericTypeInfo[Boolean],
        GenericTypeInfo[T5], GenericTypeInfo[Boolean])

    val args = Array((name0, rep0), (name1, rep1), (name2, rep2), (name3, rep3), (name4, rep4), (name5, rep5))

    val fb = new FunctionBuilder[AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]](argTypeInfo, GenericTypeInfo[R])
    var e = body
    val env = args.zipWithIndex.foldLeft(new Env[IR]()) {
      case (newEnv, ((name, rvr), i)) => newEnv.bind(name, In(i, rvr.typ))
    }
    e = Subst(e, env)
    Infer(e)
    assert(e.typ == rRep.typ)
    Emit(e, fb)
    fb.result()
  }

  def apply[T0: TypeInfo, T1: TypeInfo, T2: TypeInfo, T3: TypeInfo, T4: TypeInfo, T5: TypeInfo, R: TypeInfo](
    name0: String,
    rep0: RegionValueRep[T0],
    name1: String,
    rep1: RegionValueRep[T1],
    name2: String,
    rep2: RegionValueRep[T2],
    name3: String,
    rep3: RegionValueRep[T3],
    name4: String,
    rep4: RegionValueRep[T4],
    name5: String,
    rep5: RegionValueRep[T5],
    body: IR): (Type, () => AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]) = {
    val argTypeInfo: Array[MaybeGenericTypeInfo[_]] =
      Array(GenericTypeInfo[Region],
        GenericTypeInfo[T0], GenericTypeInfo[Boolean],
        GenericTypeInfo[T1], GenericTypeInfo[Boolean],
        GenericTypeInfo[T2], GenericTypeInfo[Boolean],
        GenericTypeInfo[T3], GenericTypeInfo[Boolean],
        GenericTypeInfo[T4], GenericTypeInfo[Boolean],
        GenericTypeInfo[T5], GenericTypeInfo[Boolean])

    val args = Array((name0, rep0), (name1, rep1), (name2, rep2), (name3, rep3), (name4, rep4), (name5, rep5))

    val fb = new FunctionBuilder[AsmFunction13[Region, T0, Boolean, T1, Boolean, T2, Boolean, T3, Boolean, T4, Boolean, T5, Boolean, R]](argTypeInfo, GenericTypeInfo[R])
    var e = body
    val env = args.zipWithIndex.foldLeft(new Env[IR]()) {
      case (newEnv, ((name, rvr), i)) => newEnv.bind(name, In(i, rvr.typ))
    }
    e = Optimize(Subst(e, env))
    Infer(e)
    assert(TypeToIRIntermediateTypeInfo(e.typ) == typeInfo[R])
    Emit(e, fb)
    i += 1
    (e.typ, fb.result(Some(new java.io.PrintWriter(new FileOutputStream(s"/Users/wang/data/compileout-$i.txt")))))
  }
}

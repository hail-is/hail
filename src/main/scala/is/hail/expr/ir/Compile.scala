package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s.{AsmFunction5, FunctionBuilder, TypeInfo}
import is.hail.expr.typ.Type

import scala.reflect.{ClassTag, classTag}

case class RegionValueRep[RVT: ClassTag](typ: Type) {
  assert(TypeToIRIntermediateClassTag(typ) == classTag[RVT])
}

object Compile {
  def apply[T0: TypeInfo, T1: TypeInfo, R: TypeInfo](name0: String, rep0: RegionValueRep[T0], name1: String, rep1: RegionValueRep[T1], rRep: RegionValueRep[R], body: IR): () => AsmFunction5[Region, T0, Boolean, T1, Boolean, R] = {
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
}

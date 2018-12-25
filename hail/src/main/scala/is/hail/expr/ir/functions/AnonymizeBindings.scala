package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types.virtual.TArray

object AnonymizeBindings {
  def apply(ir: BaseIR): BaseIR = {
    MapIR.mapBaseIR(ir, {
      case x@Let(name, value, body) =>
        val uid = genUID()
        Let(uid, value, Subst(body, Env(name -> Ref(uid, value.typ))))
      case ArrayMap(a, name, body) =>
        val uid = genUID()
        ArrayMap(a, uid, Subst(body, Env(name -> Ref(uid, a.typ.asInstanceOf[TArray].elementType))))
      case ArrayFilter(a, name, cond) =>
        val uid = genUID()
        ArrayFilter(a, uid, Subst(cond, Env(name -> Ref(uid, a.typ.asInstanceOf[TArray].elementType))))
      case ArrayFlatMap(a, name, body) =>
        val uid = genUID()
        ArrayFlatMap(a, uid, Subst(body, Env(name -> Ref(uid, a.typ.asInstanceOf[TArray].elementType))))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val uid1 = genUID()
        val uid2 = genUID()
        val env = Env[IR](accumName -> Ref(uid1, zero.typ), valueName -> Ref(uid2, a.typ.asInstanceOf[TArray].elementType))
        ArrayFold(a, zero, uid1, uid2, Subst(body, env))
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val uid1 = genUID()
        val uid2 = genUID()
        val env = Env[IR](accumName -> Ref(uid1, zero.typ), valueName -> Ref(uid2, a.typ.asInstanceOf[TArray].elementType))
        ArrayScan(a, zero, uid1, uid2, Subst(body, env))
      case ArrayFor(a, name, body) =>
        val uid = genUID()
        ArrayFor(a, uid, Subst(body, Env(name -> Ref(uid, a.typ.asInstanceOf[TArray].elementType))))
      case x => x
    })
  }
}

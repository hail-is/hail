package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types
import is.hail.expr.types._

object DictFunctions extends RegistryFunctions {

  def contains(dict: IR, key: IR) =
    If(IsNA(dict),
      NA(TBoolean()),
      ApplyComparisonOp(
        EQWithNA(key.typ),
        GetField(ArrayRef(ToArray(dict), SearchOrderedCollection(dict, key, onKey=true)), "key"),
        key))

  def get(dict: IR, key: IR, default: IR): IR = {
    val i = Ref(genUID(), TInt32())
    Let(
      i.name,
      SearchOrderedCollection(dict, key, onKey=true),
      If(IsNA(dict),
        NA(default.typ),
        If(ApplyComparisonOp(EQWithNA(key.typ), GetField(ArrayRef(ToArray(dict), i), "key"), key),
          GetField(ArrayRef(ToArray(dict), i), "value"),
          default)))
  }

  def registerAll() {
    registerIR("toDict", TArray(tv("T")))(ToDict)

    registerIR("size", TDict(tv("T"), tv("U"))) { d =>
      ArrayLen(ToArray(d))
    }

    registerIR("isEmpty", TDict(tv("T"), tv("U"))) { d =>
      ArrayFunctions.isEmpty(ToArray(d))
    }

    registerIR("contains", TDict(tv("K"), tv("V")), tv("K"))(contains)

    registerIR("get", TDict(tv("K"), tv("V")), tv("K"), tv("V"))(get)

    registerIR("get", TDict(tv("K"), tv("V")), tv("K")){ (d, k) => get(d, k, NA(types.coerce[TDict](d.typ).valueType)) }
  }
}

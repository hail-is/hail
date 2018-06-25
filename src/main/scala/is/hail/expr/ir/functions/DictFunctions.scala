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
        GetField(ArrayRef(ToArray(dict), LowerBoundOnOrderedCollection(dict, key, onKey=true)), "key"),
        key))

  def get(dict: IR, key: IR, default: IR): IR = {
    val i = Ref(genUID(), TInt32())
    Let(
      i.name,
      LowerBoundOnOrderedCollection(dict, key, onKey=true),
      If(IsNA(dict),
        NA(default.typ),
        If(ApplyComparisonOp(EQ(key.typ), GetField(ArrayRef(ToArray(dict), i), "key"), key),
          GetField(ArrayRef(ToArray(dict), i), "value"),
          default)))
  }

  val tdict = TDict(tv("key"), tv("value"))

  def registerAll() {
    registerIR("size", tdict) { d =>
      ArrayLen(ToArray(d))
    }

    registerIR("isEmpty", tdict) { d =>
      ArrayFunctions.isEmpty(ToArray(d))
    }

    registerIR("contains", tdict, tv("key"))(contains)

    registerIR("get", tdict, tv("key"), tv("value"))(get)
    registerIR("get", tdict, tv("key")) { (d, k) =>
      get(d, k, NA(types.coerce[TDict](d.typ).valueType))
    }

    registerIR("[]", tdict, tv("key")) { (d, k) =>
      val vtype = types.coerce[TBaseStruct](types.coerce[TContainer](d.typ).elementType).types(1)
      val errormsg = "Key not found in dictionary."
      get(d, k, Die(errormsg, vtype))
    }

    registerIR("dictToArray", tdict) { d =>
      val elt = Ref(genUID(), -types.coerce[TContainer](d.typ).elementType)
      ArrayMap(
        ToArray(d),
        elt.name,
        MakeTuple(Seq(GetField(elt, "key"), GetField(elt, "value"))))
    }

    registerIR("keySet", tdict) { d =>
      val pairs = Ref(genUID(), -types.coerce[TContainer](d.typ).elementType)
      ToSet(ArrayMap(ToArray(d), pairs.name, GetField(pairs, "key")))
    }

    registerIR("dict", TSet(TTuple(tv("key"), tv("value"))))(s => ToDict(ToArray(s)))

    registerIR("dict", TArray(TTuple(tv("key"), tv("value"))))(ToDict)

    registerIR("keys", tdict) { d =>
      val elt = Ref(genUID(), -types.coerce[TContainer](d.typ).elementType)
      ArrayMap(ToArray(d), elt.name, GetField(elt, "key"))
    }

    registerIR("values", tdict) { d =>
      val elt = Ref(genUID(), -types.coerce[TContainer](d.typ).elementType)
      ArrayMap(ToArray(d), elt.name, GetField(elt, "value"))
    }
  }
}

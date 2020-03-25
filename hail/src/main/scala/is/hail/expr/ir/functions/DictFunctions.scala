package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types
import is.hail.expr.types._
import is.hail.expr.types.virtual._

object DictFunctions extends RegistryFunctions {
  def contains(dict: IR, key: IR) = {
    val i = Ref(genUID(), TInt32)

    If(IsNA(dict),
      NA(TBoolean),
      Let(i.name,
        LowerBoundOnOrderedCollection(dict, key, onKey = true),
        If(i.ceq(ArrayLen(CastToArray(dict))),
          False(),
          ApplyComparisonOp(
            EQWithNA(key.typ),
            GetField(ArrayRef(CastToArray(dict), i), "key"),
            key))))
  }

  def get(dict: IR, key: IR, default: IR): IR = {
    val i = Ref(genUID(), TInt32)

    If(IsNA(dict),
      NA(default.typ),
      Let(i.name,
        LowerBoundOnOrderedCollection(dict, key, onKey=true),
        If(i.ceq(ArrayLen(CastToArray(dict))),
          default,
          If(ApplyComparisonOp(EQWithNA(key.typ), GetField(ArrayRef(CastToArray(dict), i), "key"), key),
            GetField(ArrayRef(CastToArray(dict), i), "value"),
            default))))
  }

  val tdict = TDict(tv("key"), tv("value"))

  def registerAll() {
    registerIR("isEmpty", tdict, TBoolean) { d =>
      ArrayFunctions.isEmpty(CastToArray(d))
    }

    registerIR("contains", tdict, tv("key"), TBoolean)(contains)

    registerIR("get", tdict, tv("key"), tv("value"), tv("value"))(get)
    registerIR("get", tdict, tv("key"), tv("tvalue")) { (d, k) =>
      get(d, k, NA(types.coerce[TDict](d.typ).valueType))
    }

    registerIR("index", tdict, tv("key"), tv("value")) { (d, k) =>
      val vtype = types.coerce[TBaseStruct](types.coerce[TContainer](d.typ).elementType).types(1)
      val errormsg = invoke("concat", TString,
        Str("Key '"),
        invoke("concat", TString,
          invoke("str", TString, k),
          invoke("concat", TString,
            Str("'    not found in dictionary. Keys: "),
            invoke("str", TString, invoke("keys", TArray(k.typ), d)))))
      get(d, k, Die(errormsg, vtype))
    }

    registerIR("dictToArray", tdict, TArray(TStruct("key" -> tv("key"), "value" -> tv("value")))) { d =>
      val elt = Ref(genUID(), types.coerce[TContainer](d.typ).elementType)
      ToArray(StreamMap(
        ToStream(d),
        elt.name,
        MakeTuple.ordered(Seq(GetField(elt, "key"), GetField(elt, "value")))))
    }

    registerIR("keySet", tdict, TSet(tv("key"))) { d =>
      val pairs = Ref(genUID(), types.coerce[TContainer](d.typ).elementType)
      ToSet(StreamMap(ToStream(d), pairs.name, GetField(pairs, "key")))
    }

    registerIR("dict", TSet(TTuple(tv("key"), tv("value"))), tdict)(s => ToDict(ToStream(s)))

    registerIR("dict", TArray(TTuple(tv("key"), tv("value"))), tdict)(a => ToDict(ToStream(a)))

    registerIR("keys", tdict, TArray(tv("key"))) { d =>
      val elt = Ref(genUID(), types.coerce[TContainer](d.typ).elementType)
      ToArray(StreamMap(ToStream(d), elt.name, GetField(elt, "key")))
    }

    registerIR("values", tdict, TArray(tv("value"))) { d =>
      val elt = Ref(genUID(), types.coerce[TContainer](d.typ).elementType)
      ToArray(StreamMap(ToStream(d), elt.name, GetField(elt, "value")))
    }
  }
}

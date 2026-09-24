package is.hail.expr.ir.functions

import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.defs._
import is.hail.types
import is.hail.types.virtual._

object DictFunctions extends RegistryFunctions {
  def contains(dict: Atom, key: Atom): IR =
    guardIR(!IsNA(dict)) {
      M.eval {
        for {
          i <- LowerBoundOnOrderedCollection(dict, key, onKey = true)
          a <- CastToArray(dict)
        } yield (i < ArrayLen(a)) && (GetField(ArrayRef(a, i), "key") ceq key)
      }
    }

  def get(dict: Atom, key: Atom, default: IR): IR =
    guardIR(!IsNA(dict)) {
      bindIRs(LowerBoundOnOrderedCollection(dict, key, onKey = true), CastToArray(dict)) {
        case Seq(i, arr) =>
          If(
            i ceq ArrayLen(arr),
            default.deepCopy,
            bindIR(ArrayRef(arr, i)) { elem =>
              If(
                GetField(elem, "key") ceq key,
                GetField(elem, "value"),
                default.deepCopy,
              )
            },
          )
      }
    }

  val tdict = TDict(tv("key"), tv("value"))

  override def registerAll(): Unit = {
    registerIR1("isEmpty", tdict, TBoolean)((_, d, _) => ArrayFunctions.isEmpty(CastToArray(d)))

    registerIR2("contains", tdict, tv("key"), TBoolean)((_, a, b, _) => contains(a, b))

    registerIR3("get", tdict, tv("key"), tv("value"), tv("value"))((_, a, b, c, _) => get(a, b, c))
    registerIR2("get", tdict, tv("key"), tv("tvalue")) { (_, d, k, _) =>
      get(d, k, NA(types.tcoerce[TDict](d.typ).valueType))
    }

    registerIR2("index", tdict, tv("key"), tv("value")) { (_, d, k, errorID) =>
      val vtype = types.tcoerce[TBaseStruct](types.tcoerce[TContainer](d.typ).elementType).types(1)
      val errormsg = invoke(
        "concat",
        TString,
        Str("Key "),
        invoke(
          "concat",
          TString,
          invoke("showStr", TString, k),
          invoke(
            "concat",
            TString,
            Str(" not found in dictionary. Keys: "),
            invoke("str", TString, invoke("keys", TArray(k.typ), d)),
          ),
        ),
      )
      get(d, k, Die(errormsg, vtype, errorID))
    }

    registerIR1("dictToArray", tdict, TArray(TTuple(tv("key"), tv("value")))) {
      (_, d, _) => mapArray(d)(elt => maketuple(GetField(elt, "key"), GetField(elt, "value")))
    }

    registerIR1("keySet", tdict, TSet(tv("key"))) { (_, d, _) =>
      ToSet(mapIR(ToStream(d))(GetField(_, "key")))
    }

    registerIR1("dict", TSet(TTuple(tv("key"), tv("value"))), tdict)((_, s, _) =>
      ToDict(ToStream(s))
    )

    registerIR1("dict", TArray(TTuple(tv("key"), tv("value"))), tdict)((_, a, _) =>
      ToDict(ToStream(a))
    )

    registerIR1("keys", tdict, TArray(tv("key")))((_, d, _) => mapArray(d)(GetField(_, "key")))

    registerIR1("values", tdict, TArray(tv("value"))) { (_, d, _) =>
      mapArray(d)(GetField(_, "value"))
    }
  }
}

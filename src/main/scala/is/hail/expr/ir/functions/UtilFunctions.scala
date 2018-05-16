package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.expr.types.coerce

object UtilFunctions extends RegistryFunctions {

  def parseBoolean(s: String): Boolean = s.toBoolean

  def registerAll() {
    val thisClass = getClass

    registerCode("triangle", TInt32(), TInt32()) { (_, n: Code[Int]) => (n * (n + 1)) / 2 }

    registerIR("isDefined", tv("T")) { a => ApplyUnaryPrimOp(Bang(), IsNA(a)) }
    registerIR("isMissing", tv("T")) { a => IsNA(a) }

    registerIR("[]", tv("T", _.isInstanceOf[TTuple]), TInt32()) { (a, i) => GetTupleElement(a, i.asInstanceOf[I32].x) }

    registerIR("range", TInt32(), TInt32(), TInt32())(ArrayRange)

    registerIR("range", TInt32(), TInt32())(ArrayRange(_, _, I32(1)))

    registerIR("range", TInt32())(ArrayRange(I32(0), _, I32(1)))

    registerIR("annotate", tv("T", _.isInstanceOf[TStruct]), tv("U", _.isInstanceOf[TStruct])) { (s, annotations) =>
      InsertFields(s, annotations.asInstanceOf[MakeStruct].fields)
    }

    registerCode("toInt32", TBoolean(), TInt32()) { (_, x: Code[Boolean]) => x.toI }
    registerCode("toInt64", TBoolean(), TInt64()) { (_, x: Code[Boolean]) => x.toI.toL }
    registerCode("toFloat32", TBoolean(), TFloat32()) { (_, x: Code[Boolean]) => x.toI.toF }
    registerCode("toFloat64", TBoolean(), TFloat64()) { (_, x: Code[Boolean]) => x.toI.toD }
    registerCode("toInt32", TString(), TInt32()) { (mb, x: Code[Long]) =>
      val s = asm4s.coerce[String](wrapArg(mb, TString())(x))
      Code.invokeStatic[java.lang.Integer, String, Int]("parseInt", s)
    }
    registerCode("toInt64", TString(), TInt64()) { (mb, x: Code[Long]) =>
      val s = asm4s.coerce[String](wrapArg(mb, TString())(x))
      Code.invokeStatic[java.lang.Long, String, Long]("parseLong", s)
    }
    registerCode("toFloat32", TString(), TFloat32()) { (mb, x: Code[Long]) =>
      val s = asm4s.coerce[String](wrapArg(mb, TString())(x))
      Code.invokeStatic[java.lang.Float, String, Float]("parseFloat", s)
    }
    registerCode("toFloat64", TString(), TFloat64()) { (mb, x: Code[Long]) =>
      val s = asm4s.coerce[String](wrapArg(mb, TString())(x))
      Code.invokeStatic[java.lang.Double, String, Double]("parseDouble", s)
    }
    registerCode("toBoolean", TString(), TBoolean()) { (mb, x: Code[Long]) =>
      val s = asm4s.coerce[String](wrapArg(mb, TString())(x))
      Code.invokeScalaObject[String, Boolean](thisClass, "parseBoolean", s)
    }

    val compareOps = Array(
      ("==", CodeOrdering.equiv),
      ("<", CodeOrdering.lt),
      ("<=", CodeOrdering.lteq),
      (">", CodeOrdering.gt),
      (">=", CodeOrdering.gteq))
    for ((sym, op) <- compareOps) {
      registerCodeWithMissingness(sym, tv("T"), tv("T"), TBoolean()) { case (mb, a, b) =>
        val t = tv("T").t
        val cop = mb.getCodeOrdering[Boolean](t, op, missingGreatest = true)
        val am = mb.newLocal[Boolean]
        val bm = mb.newLocal[Boolean]
        val av = mb.newLocal(typeToTypeInfo(t))
        val bv = mb.newLocal(typeToTypeInfo(t))
        val v = Code(
          am := a.m, bm := b.m, av.storeAny(a.v), bv.storeAny(b.v),
          cop(mb.getArg[Region](1), (am, av), mb.getArg[Region](1), (bm, bv)))
        EmitTriplet(Code(a.setup, b.setup), const(false), v)
      }
    }

    registerIR("!=", tv("T"), tv("T")) { (a, b) => ApplyUnaryPrimOp(Bang(), ApplySpecial("==", FastSeq(a, b))) }
  }
}

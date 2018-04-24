package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.expr.types.coerce

object UtilFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("triangle", TInt32(), TInt32()) { case (_, n: Code[Int]) => n * (n + 1) / 2 }

    registerIR("size", TArray(tv("T")))(ArrayLen)

    registerIR("sum", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val sum = genUID()
      val v = genUID()
      val zero = Cast(I64(0), t)
      ArrayFold(a, zero, sum, v, If(IsNA(Ref(v, t)), Ref(sum, t), ApplyBinaryPrimOp(Add(), Ref(sum, t), Ref(v, t))))
    }

    registerIR("*", TArray(tnum("T")), tv("T")){ (a, c) =>
      val imul = genUID()
      ArrayMap(a, imul, ApplyBinaryPrimOp(Multiply(), Ref(imul, c.typ), c))
    }

    registerIR("/", TArray(tnum("T")), tv("T")){ (a, c) =>
      val idiv = genUID()
      ArrayMap(a, idiv, ApplyBinaryPrimOp(FloatingPointDivide(), Ref(idiv, c.typ), c))
    }

    registerIR("sum", TAggregable(tnum("T")))(ApplyAggOp(_, Sum(), FastSeq()))

    registerIR("product", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val product = genUID()
      val v = genUID()
      val one = Cast(I64(1), t)
      ArrayFold(a, one, product, v, If(IsNA(Ref(v, t)), Ref(product, t), ApplyBinaryPrimOp(Multiply(), Ref(product, t), Ref(v, t))))
    }

    registerIR("count", TAggregable(tv("T"))) { agg =>
      val uid = genUID()
      ApplyAggOp(AggMap(agg, uid, I32(0)), Count(), Seq())
    }

    registerIR("min", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val min = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(min, t)),
        Ref(value, t),
        If(IsNA(Ref(value, t)),
          Ref(min, t),
          If(ApplyBinaryPrimOp(LT(), Ref(value, t), Ref(min, t)), Ref(value, t), Ref(min, t))))
      ArrayFold(a, NA(tnum("T").t), min, value, body)
    }

    registerIR("max", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val max = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(max, t)),
        Ref(value, t),
        If(IsNA(Ref(value, t)),
          Ref(max, t),
          If(ApplyBinaryPrimOp(GT(), Ref(value, t), Ref(max, t)), Ref(value, t), Ref(max, t))))
      ArrayFold(a, NA(tnum("T").t), max, value, body)
    }

    registerIR("isDefined", tv("T")) { a => ApplyUnaryPrimOp(Bang(), IsNA(a)) }
    registerIR("isMissing", tv("T")) { a => IsNA(a) }

    registerIR("[]", TArray(tv("T")), TInt32()) { (a, i) => ArrayRef(a, i) }

    registerIR("[:]", TArray(tv("T"))) { (a) => a }

    registerIR("[*:]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(If(ApplyBinaryPrimOp(LT(), i, I32(0)),
          ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
          i),
          ArrayLen(a),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("[:*]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(I32(0),
          If(ApplyBinaryPrimOp(LT(), i, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
            i),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("[*:*]", TArray(tv("T")), TInt32(), TInt32()) { (a, i, j) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(
          If(ApplyBinaryPrimOp(LT(), i, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
            i),
          If(ApplyBinaryPrimOp(LT(), j, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), j),
            j),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("[]", tv("T", _.isInstanceOf[TTuple]), TInt32()) { (a, i) => GetTupleElement(a, i.asInstanceOf[I32].x) }

    registerIR("[:]", TString()) { (a) => a }

    registerIR("range", TInt32(), TInt32(), TInt32())(ArrayRange)

    registerIR("range", TInt32(), TInt32())(ArrayRange(_, _, I32(1)))

    registerIR("range", TInt32())(ArrayRange(I32(0), _, I32(1)))

    registerIR("annotate", tv("T", _.isInstanceOf[TStruct]), tv("U", _.isInstanceOf[TStruct])) { (s, annotations) =>
      InsertFields(s, annotations.asInstanceOf[MakeStruct].fields)
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
          am := a.m, bm := b.m, av := a.v, bv := b.v,
          cop(mb.getArg[Region](1), (am, av), mb.getArg[Region](1), (bm, bv)))
        EmitTriplet(Code(a.setup, b.setup), const(false), v)
      }
    }
  }
}

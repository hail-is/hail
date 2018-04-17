package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.expr.types.coerce
import is.hail.asm4s

object UtilFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("triangle", TInt32(), TInt32()) { (_, n: Code[Int]) => n * (n + 1) / 2 }

    registerIR("size", TArray(tv("T")))(ArrayLen)

    registerIR("sum", TArray(tnum("T"))) { a =>
      val zero = Literal(0, coerce[TArray](a.typ).elementType)
      ArrayFold(a, zero, "sum", "v", If(IsNA(Ref("v")), Ref("sum"), ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))))
    }

    registerIR("*", TArray(tnum("T")), tv("T")){ (a, c) => ArrayMap(a, "imul", ApplyBinaryPrimOp(Multiply(), Ref("imul"), c)) }

    registerIR("sum", TAggregable(tnum("T")))(ApplyAggOp(_, Sum(), FastSeq()))

    registerIR("product", TArray(tnum("T"))) { a =>
      val product = genUID()
      val v = genUID()
      val one = Literal(1, coerce[TArray](a.typ).elementType)
      ArrayFold(a, one, product, v, If(IsNA(Ref(v)), Ref(product), ApplyBinaryPrimOp(Multiply(), Ref(product), Ref(v))))
    }

    registerIR("count", TAggregable(tv("T"))) { agg =>
      val uid = genUID()
      ApplyAggOp(AggMap(agg, uid, I32(0)), Count(), Seq())
    }

    registerIR("min", TArray(tnum("T"))) { a =>
      val min = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(min)),
        Ref(value),
        If(IsNA(Ref(value)),
          Ref(min),
          If(ApplyBinaryPrimOp(LT(), Ref(value), Ref(min)), Ref(value), Ref(min))))
      ArrayFold(a, NA(tnum("T").t), min, value, body)
    }

    registerIR("max", TArray(tnum("T"))) { a =>
      val max = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(max)),
        Ref(value),
        If(IsNA(Ref(value)),
          Ref(max),
          If(ApplyBinaryPrimOp(GT(), Ref(value), Ref(max)), Ref(value), Ref(max))))
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
        ArrayRef(a, Ref(idx)))
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
        ArrayRef(a, Ref(idx)))
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
        ArrayRef(a, Ref(idx)))
    }

    registerIR("[]", tv("T", _.isInstanceOf[TTuple]), TInt32()) { (a, i) => GetTupleElement(a, i.asInstanceOf[I32].x) }

    registerIR("[:]", TString()) { (a) => a }

    registerIR("range", TInt32(), TInt32(), TInt32())(ArrayRange)

    registerIR("range", TInt32(), TInt32())(ArrayRange(_, _, I32(1)))

    registerIR("range", TInt32())(ArrayRange(I32(0), _, I32(1)))

    registerIR("annotate", tv("T", _.isInstanceOf[TStruct]), tv("U", _.isInstanceOf[TStruct])) { (s, annotations) =>
      InsertFields(s, annotations.asInstanceOf[MakeStruct].fields)
    }

    registerCodeWithMissingness("==", tv("T"), tv("T"), TBoolean()) { (mb, v1, v2) =>
      val tactual = tv("T").t match {
        case tc: ComplexType => tc.representation
        case t => t
      }
      val comparison = (tactual, v1.v, v2.v) match {
        case (_: TBoolean, x1: Code[Boolean], x2: Code[Boolean]) => x1.ceq(x2)
        case (_: TInt32, x1: Code[Int], x2: Code[Int]) => x1.ceq(x2)
        case (_: TInt64, x1: Code[Long], x2: Code[Long]) => x1.ceq(x2)
        case (_: TFloat32, x1: Code[Float], x2: Code[Float]) => x1.ceq(x2)
        case (_: TFloat64, x1: Code[Double], x2: Code[Double]) => x1.ceq(x2)
        case (t, o1: Code[Long], o2: Code[Long]) =>
          val ord = CodeOrdering(t, missingGreatest = true)
          ord.compare(mb, mb.getArg[Region](1), o1, mb.getArg[Region](1), o2).ceq(0)
        case (t, x1, x2) =>
          throw new UnsupportedOperationException(s"can't compare things of type $t with classes ${ x1.getClass } and ${ x2.getClass }")
      }

      val setup = Code(v1.setup, v2.setup)
      val v1m = mb.newLocal[Boolean]
      val vout = Code(
        v1m := v1.m,
        v1m.cne(v2.m).mux(const(false), v1m.mux(const(true), comparison)))
      EmitTriplet(setup, const(false), vout)
    }

    registerCodeWithMissingness("!=", tv("T"), tv("T"), TBoolean()) { (mb, v1, v2) =>
      val tactual = tv("T").t match {
        case tc: ComplexType => tc.representation
        case t => t
      }
      val comparison = (tactual, v1.v, v2.v) match {
        case (_: TBoolean, x1: Code[Boolean], x2: Code[Boolean]) => x1.cne(x2)
        case (_: TInt32, x1: Code[Int], x2: Code[Int]) => x1.cne(x2)
        case (_: TInt64, x1: Code[Long], x2: Code[Long]) => x1.cne(x2)
        case (_: TFloat32, x1: Code[Float], x2: Code[Float]) => x1.cne(x2)
        case (_: TFloat64, x1: Code[Double], x2: Code[Double]) => x1.cne(x2)
        case (t, o1: Code[Long], o2: Code[Long]) =>
          val ord = CodeOrdering(t, missingGreatest = true)
          ord.compare(mb, mb.getArg[Region](1), o1, mb.getArg[Region](1), o2).cne(0)
        case (t, x1, x2) =>
          throw new UnsupportedOperationException(s"can't compare things of type $t with classes ${ x1.getClass } and ${ x2.getClass }")
      }

      val setup = Code(v1.setup, v2.setup)
      val v1m = mb.newLocal[Boolean]
      val vout = Code(
        v1m := v1.m,
        v1m.cne(v2.m).mux(const(true), v1m.mux(const(false), comparison)))
      EmitTriplet(setup, const(false), vout)
    }
  }
}

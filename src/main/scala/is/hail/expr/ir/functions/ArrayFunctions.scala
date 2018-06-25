package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.coerce
import is.hail.utils.FastSeq

object ArrayFunctions extends RegistryFunctions {
  def mean(a: IR): IR = {
    val t = -coerce[TArray](a.typ).elementType
    val tAccum = TStruct("sum" -> TFloat64(), "n" -> TInt32())
    val accum = genUID()
    val v = genUID()
    val result = genUID()

    def updateAccum(sum: IR, n: IR): IR =
      MakeStruct(FastSeq("sum" -> sum, "n" -> n))

    Let(result,
    ArrayFold(
      a,
      MakeStruct(FastSeq("sum" -> F64(0), "n" -> I32(0))),
      accum,
      v,
      If(IsNA(Ref(v, t)),
        Ref(accum, tAccum),
        updateAccum(
          ApplyBinaryPrimOp(Add(), GetField(Ref(accum, tAccum), "sum"), Cast(Ref(v, t), TFloat64())),
          ApplyBinaryPrimOp(Add(), GetField(Ref(accum, tAccum), "n"), I32(1))))),
      If(ApplyComparisonOp(EQ(TInt32()), GetField(Ref(result, tAccum), "n"), I32(0)),
        NA(TFloat64()),
        ApplyBinaryPrimOp(FloatingPointDivide(),
          GetField(Ref(result, tAccum), "sum"),
          Cast(GetField(Ref(result, tAccum), "n"), TFloat64()))))
  }

  def isEmpty(a: IR): IR = ApplyComparisonOp(EQ(TInt32()), ArrayLen(a), I32(0))

  def extend(a1: IR, a2: IR): IR = {
    val uid = genUID()
    val typ = a1.typ
    If(IsNA(a1), NA(typ),
      If(IsNA(a2), NA(typ),
        ArrayFlatMap(
          MakeArray(Seq(a1, a2), TArray(typ)),
          uid,
          Ref(uid, a1.typ)
        )
      ))
  }

  def sum(a: IR): IR = {
    val t = -coerce[TArray](a.typ).elementType
    val sum = genUID()
    val v = genUID()
    val zero = Cast(I64(0), t)
    ArrayFold(a, zero, sum, v, If(IsNA(Ref(v, t)), Ref(sum, t), ApplyBinaryPrimOp(Add(), Ref(sum, t), Ref(v, t))))
  }

  def product(a: IR): IR = {
    val t = -coerce[TArray](a.typ).elementType
    val product = genUID()
    val v = genUID()
    val one = Cast(I64(1), t)
    ArrayFold(a, one, product, v, If(IsNA(Ref(v, t)), Ref(product, t), ApplyBinaryPrimOp(Multiply(), Ref(product, t), Ref(v, t))))
  }

  def registerAll() {
    registerIR("size", TArray(tv("T")))(ArrayLen)

    registerIR("length", TArray(tv("T")))(ArrayLen)

    registerIR("isEmpty", TArray(tv("T")))(isEmpty)

    registerIR("sort", TArray(tv("T")), TBoolean())(ArraySort(_, _, false))

    registerIR("sort", TArray(tv("T"))) { a =>
      ArraySort(a, True(), false)
    }

    registerIR("extend", TArray(tv("T")), TArray(tv("T")))(extend)

    registerIR("append", TArray(tv("T")), tv("T")) { (a, c) =>
      extend(a, MakeArray(Seq(c), TArray(c.typ)))
    }

    val arrayOps: Array[(String, (IR, IR) => IR)] =
      Array(
        ("*", ApplyBinaryPrimOp(Multiply(), _, _)),
        ("/", ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
        ("//", ApplyBinaryPrimOp(RoundToNegInfDivide(), _, _)),
        ("+", ApplyBinaryPrimOp(Add(), _, _)),
        ("-", ApplyBinaryPrimOp(Subtract(), _, _)),
        ("**", { (ir1: IR, ir2: IR) => Apply("**", Seq(ir1, ir2)) }),
        ("%", { (ir1: IR, ir2: IR) => Apply("%", Seq(ir1, ir2)) }))

    for ((stringOp, irOp) <- arrayOps) {
      registerIR(stringOp, TArray(tnum("T")), tv("T")) { (a, c) =>
        val i = genUID()
        ArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR(stringOp, tnum("T"), TArray(tv("T"))) { (c, a) =>
        val i = genUID()
        ArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR(stringOp, TArray(tnum("T")), TArray(tv("T"))) { (array1, array2) =>
        val a1id = genUID()
        val a1 = Ref(a1id, array1.typ)
        val a2id = genUID()
        val a2 = Ref(a2id, array2.typ)
        val iid = genUID()
        val i = Ref(iid, TInt32())
        val body =
          ArrayMap(ArrayRange(I32(0), ArrayLen(a1), I32(1)), iid,
            irOp(ArrayRef(a1, i), ArrayRef(a2, i)))
        val guarded =
          If(ApplyComparisonOp(EQ(TInt32()), ArrayLen(a1), ArrayLen(a2)),
            body,
            Die("Arrays must have same length", body.typ))
        Let(a1id, array1, Let(a2id, array2, guarded))
      }
    }

    registerIR("sum", TArray(tnum("T")))(sum)

    registerIR("product", TArray(tnum("T")))(product)

    registerIR("min", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val min = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(min, t)),
        Ref(value, t),
        If(IsNA(Ref(value, t)),
          Ref(min, t),
          If(ApplyComparisonOp(LT(t), Ref(value, t), Ref(min, t)), Ref(value, t), Ref(min, t))))
      ArrayFold(a, NA(t), min, value, body)
    }

    registerIR("max", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val max = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(max, t)),
        Ref(value, t),
        If(IsNA(Ref(value, t)),
          Ref(max, t),
          If(ApplyComparisonOp(GT(t), Ref(value, t), Ref(max, t)), Ref(value, t), Ref(max, t))))
      ArrayFold(a, NA(t), max, value, body)
    }

    registerIR("mean", TArray(tnum("T")))(mean)

    registerIR("median", TArray(tnum("T"))) { array =>
      val t = -array.typ.asInstanceOf[TArray].elementType
      val v = Ref(genUID(), t)
      val a = Ref(genUID(), TArray(t))
      val size = Ref(genUID(), TInt32())
      val lastIdx = size - 1
      val midIdx = lastIdx.floorDiv(2)
      def ref(i: IR) = ArrayRef(a, i)
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      Let(a.name, ArraySort(ArrayFilter(array, v.name, !IsNA(v)), True()),
        If(IsNA(a),
          NA(t),
          Let(size.name,
            ArrayLen(a),
            If(size.ceq(0),
              NA(t),
              If(invoke("%", size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)))))))
    }
    
    def argF(a: IR, op: (Type) => ComparisonOp): IR = {
      val t = -coerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32())
      val accum = genUID()
      val value = genUID()
      val m = genUID()
      val idx = genUID()

      def updateAccum(min: IR, midx: IR): IR =
        MakeStruct(FastSeq("m" -> min, "midx" -> midx))

      val body =
        Let(value, ArrayRef(a, Ref(idx, TInt32())),
          Let(m, GetField(Ref(accum, tAccum), "m"),
            If(IsNA(Ref(value, t)),
              Ref(accum, tAccum),
              If(IsNA(Ref(m, t)),
                updateAccum(Ref(value, t), Ref(idx, TInt32())),
                If(ApplyComparisonOp(op(t), Ref(value, t), Ref(m, t)),
                  updateAccum(Ref(value, t), Ref(idx, TInt32())),
                  Ref(accum, tAccum))))))

        GetField(ArrayFold(
          ArrayRange(I32(0), ArrayLen(a), I32(1)),
          NA(tAccum),
          accum,
          idx,
          body
        ), "midx")
    }

    registerIR("argmin", TArray(tv("T")))(argF(_, LT(_)))

    registerIR("argmax", TArray(tv("T")))(argF(_, GT(_)))

    def uniqueIndex(a: IR, op: (Type) => ComparisonOp): IR = {
      val t = -coerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32(), "count" -> TInt32())
      val accum = genUID()
      val value = genUID()
      val m = genUID()
      val idx = genUID()
      val result = genUID()

      def updateAccum(m: IR, midx: IR, count: IR): IR =
        MakeStruct(FastSeq("m" -> m, "midx" -> midx, "count" -> count))

      val body =
        Let(value, ArrayRef(a, Ref(idx, TInt32())),
          Let(m, GetField(Ref(accum, tAccum), "m"),
            If(IsNA(Ref(value, t)),
              Ref(accum, tAccum),
              If(IsNA(Ref(m, t)),
                updateAccum(Ref(value, t), Ref(idx, TInt32()), I32(1)),
                If(ApplyComparisonOp(op(t), Ref(value, t), Ref(m, t)),
                  updateAccum(Ref(value, t), Ref(idx, TInt32()), I32(1)),
                  If(ApplyComparisonOp(EQ(t), Ref(value, t), Ref(m, t)),
                    updateAccum(Ref(value, t), Ref(idx, TInt32()), ApplyBinaryPrimOp(Add(), GetField(Ref(accum, tAccum), "count"), I32(1))),
                    Ref(accum, tAccum)))))))

      Let(result, ArrayFold(
        ArrayRange(I32(0), ArrayLen(a), I32(1)),
        NA(tAccum),
        accum,
        idx,
        body
      ), If(ApplyComparisonOp(EQ(TInt32()), GetField(Ref(result, tAccum), "count"), I32(1)),
        GetField(Ref(result, tAccum), "midx"),
        NA(TInt32())))
    }

    registerIR("uniqueMinIndex", TArray(tv("T")))(uniqueIndex(_, LT(_)))

    registerIR("uniqueMaxIndex", TArray(tv("T")))(uniqueIndex(_, GT(_)))

    registerIR("[]", TArray(tv("T")), TInt32()) { (a, i) =>
      ArrayRef(
        a,
        If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
          ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
          i))
    }

    registerIR("[:]", TArray(tv("T"))) { (a) => a }

    registerIR("[*:]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(
          If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
            UtilFunctions.max(
              ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
              I32(0)),
            i),
          ArrayLen(a),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("[:*]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      If(IsNA(a), a,
        ArrayMap(
          ArrayRange(
            I32(0),
            If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
              ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
              UtilFunctions.min(i, ArrayLen(a))),
            I32(1)),
          idx,
          ArrayRef(a, Ref(idx, TInt32()))))
    }

    registerIR("[*:*]", TArray(tv("T")), TInt32(), TInt32()) { (a, i, j) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(
          If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
            UtilFunctions.max(
              ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
              I32(0)),
            i),
          If(ApplyComparisonOp(LT(TInt32()), j, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), j),
            UtilFunctions.min(j, ArrayLen(a))),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("flatten", TArray(tv("T"))) { a =>
      val elt = Ref(genUID(), coerce[TArray](a.typ).elementType)
      ArrayFlatMap(a, elt.name, elt)
    }
  }
}

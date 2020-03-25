package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.coerce
import is.hail.expr.types.physical.{PArray, PFloat64}
import is.hail.expr.types.virtual._
import is.hail.utils._

object ArrayFunctions extends RegistryFunctions {
  val arrayOps: Array[(String, Type, Type, (IR, IR) => IR)] =
    Array(
      ("mul", tnum("T"), tv("T"), ApplyBinaryPrimOp(Multiply(), _, _)),
      ("div", TInt32, TFloat32, ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("div", TInt64, TFloat32, ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("div", TFloat32, TFloat32, ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("div", TFloat64, TFloat64, ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("floordiv", tnum("T"), tv("T"), ApplyBinaryPrimOp(RoundToNegInfDivide(), _, _)),
      ("add", tnum("T"), tv("T"), ApplyBinaryPrimOp(Add(), _, _)),
      ("sub", tnum("T"), tv("T"), ApplyBinaryPrimOp(Subtract(), _, _)),
      ("pow", tnum("T"), TFloat64, (ir1: IR, ir2: IR) => Apply("pow", Seq(ir1, ir2), TFloat64)),
      ("mod", tnum("T"), tv("T"), (ir1: IR, ir2: IR) => Apply("mod", Seq(ir1, ir2), ir2.typ)))

  def mean(args: Seq[IR]): IR = {
    val Seq(a) = args
    val t = coerce[TArray](a.typ).elementType
    val elt = genUID()
    val n = genUID()
    val sum = genUID()
    StreamFold2(
      ToStream(a),
      FastIndexedSeq((n, I32(0)), (sum, zero(t))),
      elt,
      FastIndexedSeq(Ref(n, TInt32) + I32(1), Ref(sum, t) + Ref(elt, t)),
      Cast(Ref(sum, t), TFloat64) / Cast(Ref(n, TInt32), TFloat64)
    )
  }

  def isEmpty(a: IR): IR = ApplyComparisonOp(EQ(TInt32), ArrayLen(a), I32(0))

  def extend(a1: IR, a2: IR): IR = {
    val uid = genUID()
    val typ = a1.typ
    If(IsNA(a1),
      NA(typ),
      If(IsNA(a2),
        NA(typ),
        ToArray(StreamFlatMap(
          MakeStream(Seq(a1, a2), TStream(typ)),
          uid,
          ToStream(Ref(uid, a1.typ))))))
  }

  def exists(a: IR, cond: IR => IR): IR = {
    val t = coerce[TArray](a.typ).elementType
    StreamFold(
      ToStream(a),
      False(),
      "acc",
      "elt",
      invoke("lor",TBoolean,
        Ref("acc", TBoolean),
        cond(Ref("elt", t))))
  }

  def contains(a: IR, value: IR): IR = {
    exists(a, elt => ApplyComparisonOp(
      EQWithNA(elt.typ, value.typ),
      elt,
      value))
  }

  def sum(a: IR): IR = {
    val t = coerce[TArray](a.typ).elementType
    val sum = genUID()
    val v = genUID()
    val zero = Cast(I64(0), t)
    StreamFold(ToStream(a), zero, sum, v, ApplyBinaryPrimOp(Add(), Ref(sum, t), Ref(v, t)))
  }

  def product(a: IR): IR = {
    val t = coerce[TArray](a.typ).elementType
    val product = genUID()
    val v = genUID()
    val one = Cast(I64(1), t)
    StreamFold(ToStream(a), one, product, v, ApplyBinaryPrimOp(Multiply(), Ref(product, t), Ref(v, t)))
  }

  def registerAll() {
    registerIR("isEmpty", TArray(tv("T")), TBoolean)(isEmpty)

    registerIR("extend", TArray(tv("T")), TArray(tv("T")), TArray(tv("T")))(extend)

    registerIR("append", TArray(tv("T")), tv("T"), TArray(tv("T"))) { (a, c) =>
      extend(a, MakeArray(Seq(c), TArray(c.typ)))
    }

    registerIR("contains", TArray(tv("T")), tv("T"), TBoolean) { (a, e) => contains(a, e) }

    for ((stringOp, argType, retType, irOp) <- arrayOps) {
      registerIR(stringOp, TArray(argType), argType, TArray(retType)) { (a, c) =>
        val i = genUID()
        ToArray(StreamMap(ToStream(a), i, irOp(Ref(i, c.typ), c)))
      }

      registerIR(stringOp, argType, TArray(argType), TArray(retType)) { (c, a) =>
        val i = genUID()
        ToArray(StreamMap(ToStream(a), i, irOp(c, Ref(i, c.typ))))
      }

      registerIR(stringOp, TArray(argType), TArray(argType), TArray(retType)) { (array1, array2) =>
        val a1id = genUID()
        val e1 = Ref(a1id, coerce[TArray](array1.typ).elementType)
        val a2id = genUID()
        val e2 = Ref(a2id, coerce[TArray](array2.typ).elementType)
        ToArray(StreamZip(FastIndexedSeq(ToStream(array1), ToStream(array2)), FastIndexedSeq(a1id, a2id), irOp(e1, e2), ArrayZipBehavior.AssertSameLength))
      }
    }

    registerIR("sum", TArray(tnum("T")), tv("T"))(sum)

    registerIR("product", TArray(tnum("T")), tv("T"))(product)

    def makeMinMaxOp(op: String): Seq[IR] => IR = {
      { case Seq(a) =>
        val t = coerce[TArray](a.typ).elementType
        val value = genUID()
        val first = genUID()
        val acc = genUID()
        StreamFold2(ToStream(a),
          FastIndexedSeq((acc, NA(t)), (first, True())),
          value,
          FastIndexedSeq(
            If(Ref(first, TBoolean), Ref(value, t), invoke(op, t, Ref(acc, t), Ref(value, t))),
            False()
          ),
          Ref(acc, t))
      }
    }

    registerIR("min", Array(TArray(tnum("T"))), tv("T"), inline = true)(makeMinMaxOp("min"))
    registerIR("nanmin", Array(TArray(tnum("T"))), tv("T"), inline = true)(makeMinMaxOp("nanmin"))
    registerIR("max", Array(TArray(tnum("T"))), tv("T"), inline = true)(makeMinMaxOp("max"))
    registerIR("nanmax", Array(TArray(tnum("T"))), tv("T"), inline = true)(makeMinMaxOp("nanmax"))

    registerIR("mean", Array(TArray(tnum("T"))), TFloat64, inline = true)(mean)

    registerIR("median", TArray(tnum("T")), tv("T")) { array =>
      val t = array.typ.asInstanceOf[TArray].elementType
      val v = Ref(genUID(), t)
      val a = Ref(genUID(), TArray(t))
      val size = Ref(genUID(), TInt32)
      val lastIdx = size - 1
      val midIdx = lastIdx.floorDiv(2)
      def ref(i: IR) = ArrayRef(a, i)
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      Let(a.name, ArraySort(StreamFilter(ToStream(array), v.name, !IsNA(v))),
        If(IsNA(a),
          NA(t),
          Let(size.name,
            ArrayLen(a),
            If(size.ceq(0),
              NA(t),
              If(invoke("mod", TInt32, size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)))))))
    }
    
    def argF(a: IR, op: (Type) => ComparisonOp[Boolean]): IR = {
      val t = coerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32)
      val accum = genUID()
      val value = genUID()
      val m = genUID()
      val idx = genUID()

      def updateAccum(min: IR, midx: IR): IR =
        MakeStruct(FastSeq("m" -> min, "midx" -> midx))

      val body =
        Let(value, ArrayRef(a, Ref(idx, TInt32)),
          Let(m, GetField(Ref(accum, tAccum), "m"),
            If(IsNA(Ref(value, t)),
              Ref(accum, tAccum),
              If(IsNA(Ref(m, t)),
                updateAccum(Ref(value, t), Ref(idx, TInt32)),
                If(ApplyComparisonOp(op(t), Ref(value, t), Ref(m, t)),
                  updateAccum(Ref(value, t), Ref(idx, TInt32)),
                  Ref(accum, tAccum))))))
      GetField(StreamFold(
        StreamRange(I32(0), ArrayLen(a), I32(1)),
        NA(tAccum),
        accum,
        idx,
        body
      ), "midx")
    }

    registerIR("argmin", TArray(tv("T")), TInt32)(argF(_, LT(_)))

    registerIR("argmax", TArray(tv("T")), TInt32)(argF(_, GT(_)))

    def uniqueIndex(a: IR, op: (Type) => ComparisonOp[Boolean]): IR = {
      val t = coerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32, "count" -> TInt32)
      val accum = genUID()
      val value = genUID()
      val m = genUID()
      val idx = genUID()
      val result = genUID()

      def updateAccum(m: IR, midx: IR, count: IR): IR =
        MakeStruct(FastSeq("m" -> m, "midx" -> midx, "count" -> count))

      val body =
        Let(value, ArrayRef(a, Ref(idx, TInt32)),
          Let(m, GetField(Ref(accum, tAccum), "m"),
            If(IsNA(Ref(value, t)),
              Ref(accum, tAccum),
              If(IsNA(Ref(m, t)),
                updateAccum(Ref(value, t), Ref(idx, TInt32), I32(1)),
                If(ApplyComparisonOp(op(t), Ref(value, t), Ref(m, t)),
                  updateAccum(Ref(value, t), Ref(idx, TInt32), I32(1)),
                  If(ApplyComparisonOp(EQ(t), Ref(value, t), Ref(m, t)),
                    updateAccum(
                      Ref(value, t),
                      Ref(idx, TInt32),
                      ApplyBinaryPrimOp(Add(), GetField(Ref(accum, tAccum), "count"), I32(1))),
                    Ref(accum, tAccum)))))))

      Let(result, StreamFold(
        StreamRange(I32(0), ArrayLen(a), I32(1)),
        NA(tAccum),
        accum,
        idx,
        body
      ), If(ApplyComparisonOp(EQ(TInt32), GetField(Ref(result, tAccum), "count"), I32(1)),
        GetField(Ref(result, tAccum), "midx"),
        NA(TInt32)))
    }

    registerIR("uniqueMinIndex", TArray(tv("T")), TInt32)(uniqueIndex(_, LT(_)))

    registerIR("uniqueMaxIndex", TArray(tv("T")), TInt32)(uniqueIndex(_, GT(_)))

    registerIR("indexArray", TArray(tv("T")), TInt32, TString, tv("T")) { (a, i, s) =>
      ArrayRef(
        a,
        If(ApplyComparisonOp(LT(TInt32), i, I32(0)),
          ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
          i), s)
    }

    registerIR("sliceRight", TArray(tv("T")), TInt32, TArray(tv("T"))) { (a, i) =>
      val idx = genUID()
      ToArray(StreamMap(
        StreamRange(
          If(ApplyComparisonOp(LT(TInt32), i, I32(0)),
            UtilFunctions.intMax(
              ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
              I32(0)),
            i),
          ArrayLen(a),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32))))
    }

    registerIR("sliceLeft", TArray(tv("T")), TInt32, TArray(tv("T"))) { (a, i) =>
      val idx = genUID()
      If(IsNA(a), a,
        ToArray(StreamMap(
          StreamRange(
            I32(0),
            If(ApplyComparisonOp(LT(TInt32), i, I32(0)),
              ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
              UtilFunctions.intMin(i, ArrayLen(a))),
            I32(1)),
          idx,
          ArrayRef(a, Ref(idx, TInt32)))))
    }

    registerIR("slice", TArray(tv("T")), TInt32, TInt32, TArray(tv("T"))) { (a, i, j) =>
      val idx = genUID()
      ToArray(StreamMap(
        StreamRange(
          If(ApplyComparisonOp(LT(TInt32), i, I32(0)),
            UtilFunctions.intMax(
              ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
              I32(0)),
            i),
          If(ApplyComparisonOp(LT(TInt32), j, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), j),
            UtilFunctions.intMin(j, ArrayLen(a))),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32))))
    }

    registerIR("flatten", TArray(TArray(tv("T"))), TArray(tv("T"))) { a =>
      val elt = Ref(genUID(), coerce[TArray](a.typ).elementType)
      ToArray(StreamFlatMap(ToStream(a), elt.name, ToStream(elt)))
    }

    registerCodeWithMissingness("corr", TArray(TFloat64), TArray(TFloat64), TFloat64, null) {
      case (r, rt, (t1: PArray, EmitCode(setup1, m1, v1)), (t2: PArray, EmitCode(setup2, m2, v2))) =>
        val a1 = r.mb.newLocal[Long]()
        val a2 = r.mb.newLocal[Long]()
        val xSum = r.mb.newLocal[Double]()
        val ySum = r.mb.newLocal[Double]()
        val xSqSum = r.mb.newLocal[Double]()
        val ySqSum = r.mb.newLocal[Double]()
        val xySum = r.mb.newLocal[Double]()
        val n = r.mb.newLocal[Int]()
        val i = r.mb.newLocal[Int]()
        val l1 = r.mb.newLocal[Int]()
        val l2 = r.mb.newLocal[Int]()
        val x = r.mb.newLocal[Double]()
        val y = r.mb.newLocal[Double]()

        EmitCode(
          Code(
            setup1,
            setup2),
          m1 || m2 || Code(
            a1 := v1.tcode[Long],
            a2 := v2.tcode[Long],
            l1 := t1.loadLength(a1),
            l2 := t2.loadLength(a2),
            l1.cne(l2).mux(
              Code._fatal[Boolean](new CodeString("'corr': cannot compute correlation between arrays of different lengths: ")
                .concat(l1.toS)
                .concat(", ")
                .concat(l2.toS)),
              l1.ceq(0))),
          PCode(PFloat64(), Code(
            i := 0,
            n := 0,
            xSum := 0d,
            ySum := 0d,
            xSqSum := 0d,
            ySqSum := 0d,
            xySum := 0d,
            Code.whileLoop(i < l1,
              Code(
                (t1.isElementDefined(a1, i) && t2.isElementDefined(a2, i)).mux(
                  Code(
                    x := Region.loadDouble(t1.loadElement(a1, i)),
                    xSum := xSum + x,
                    xSqSum := xSqSum + x * x,
                    y := Region.loadDouble(t2.loadElement(a2, i)),
                    ySum := ySum + y,
                    ySqSum := ySqSum + y * y,
                    xySum := xySum + x * y,
                    n := n + 1
                  ),
                  Code._empty
                ),
                i := i + 1
              )
            ),
            (n.toD * xySum - xSum * ySum) / Code.invokeScalaObject[Double, Double](
              MathFunctions.mathPackageClass,
              "sqrt",
              (n.toD * xSqSum - xSum * xSum) * (n.toD * ySqSum - ySum * ySum))
          )
        ))
    }
  }
}

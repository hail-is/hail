package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical.{PCanonicalArray, PType}
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SFloat64, SInt32, SInt32Value}
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

object ArrayFunctions extends RegistryFunctions {
  val arrayOps: Array[(String, Type, Type, (IR, IR, Int) => IR)] =
    Array(
      ("mul", tnum("T"), tv("T"), (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(Multiply(), ir1, ir2)),
      (
        "div",
        TInt32,
        TFloat32,
        (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(FloatingPointDivide(), ir1, ir2),
      ),
      (
        "div",
        TInt64,
        TFloat32,
        (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(FloatingPointDivide(), ir1, ir2),
      ),
      (
        "div",
        TFloat32,
        TFloat32,
        (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(FloatingPointDivide(), ir1, ir2),
      ),
      (
        "div",
        TFloat64,
        TFloat64,
        (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(FloatingPointDivide(), ir1, ir2),
      ),
      (
        "floordiv",
        tnum("T"),
        tv("T"),
        (ir1: IR, ir2: IR, _) =>
          ApplyBinaryPrimOp(RoundToNegInfDivide(), ir1, ir2),
      ),
      ("add", tnum("T"), tv("T"), (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(Add(), ir1, ir2)),
      ("sub", tnum("T"), tv("T"), (ir1: IR, ir2: IR, _) => ApplyBinaryPrimOp(Subtract(), ir1, ir2)),
      (
        "pow",
        tnum("T"),
        TFloat64,
        (ir1: IR, ir2: IR, errorID: Int) =>
          Apply("pow", Seq(), Seq(ir1, ir2), TFloat64, errorID),
      ),
      (
        "mod",
        tnum("T"),
        tv("T"),
        (ir1: IR, ir2: IR, errorID: Int) =>
          Apply("mod", Seq(), Seq(ir1, ir2), ir2.typ, errorID),
      ),
    )

  def mean(args: Seq[IR]): IR = {
    val Seq(a) = args
    val t = tcoerce[TArray](a.typ).elementType
    val elt = genUID()
    val n = genUID()
    val sum = genUID()
    StreamFold2(
      ToStream(a),
      FastSeq((n, I32(0)), (sum, zero(t))),
      elt,
      FastSeq(Ref(n, TInt32) + I32(1), Ref(sum, t) + Ref(elt, t)),
      Cast(Ref(sum, t), TFloat64) / Cast(Ref(n, TInt32), TFloat64),
    )
  }

  def isEmpty(a: IR): IR = ApplyComparisonOp(EQ(TInt32), ArrayLen(a), I32(0))

  def extend(a1: IR, a2: IR): IR = {
    val uid = genUID()
    val typ = a1.typ
    If(
      IsNA(a1),
      NA(typ),
      If(
        IsNA(a2),
        NA(typ),
        ToArray(StreamFlatMap(
          MakeStream(FastSeq(a1, a2), TStream(typ)),
          uid,
          ToStream(Ref(uid, a1.typ)),
        )),
      ),
    )
  }

  def exists(a: IR, cond: IR => IR): IR = {
    val t = tcoerce[TArray](a.typ).elementType
    StreamFold(
      ToStream(a),
      False(),
      "acc",
      "elt",
      invoke("lor", TBoolean, Ref("acc", TBoolean), cond(Ref("elt", t))),
    )
  }

  def contains(a: IR, value: IR): IR =
    exists(
      a,
      elt =>
        ApplyComparisonOp(
          EQWithNA(elt.typ, value.typ),
          elt,
          value,
        ),
    )

  def sum(a: IR): IR = {
    val t = tcoerce[TArray](a.typ).elementType
    val sum = genUID()
    val v = genUID()
    val zero = Cast(I64(0), t)
    StreamFold(ToStream(a), zero, sum, v, ApplyBinaryPrimOp(Add(), Ref(sum, t), Ref(v, t)))
  }

  def product(a: IR): IR = {
    val t = tcoerce[TArray](a.typ).elementType
    val product = genUID()
    val v = genUID()
    val one = Cast(I64(1), t)
    StreamFold(
      ToStream(a),
      one,
      product,
      v,
      ApplyBinaryPrimOp(Multiply(), Ref(product, t), Ref(v, t)),
    )
  }

  def registerAll() {
    registerIR1("isEmpty", TArray(tv("T")), TBoolean)((_, a, _) => isEmpty(a))

    registerIR2("extend", TArray(tv("T")), TArray(tv("T")), TArray(tv("T")))((_, a, b, _) =>
      extend(a, b)
    )

    registerIR2("append", TArray(tv("T")), tv("T"), TArray(tv("T"))) { (_, a, c, _) =>
      extend(a, MakeArray(FastSeq(c), TArray(c.typ)))
    }

    registerIR2("contains", TArray(tv("T")), tv("T"), TBoolean)((_, a, e, _) => contains(a, e))

    for ((stringOp, argType, retType, irOp) <- arrayOps) {
      registerIR2(stringOp, TArray(argType), argType, TArray(retType)) { (_, a, c, errorID) =>
        val i = genUID()
        ToArray(StreamMap(ToStream(a), i, irOp(Ref(i, c.typ), c, errorID)))
      }

      registerIR2(stringOp, argType, TArray(argType), TArray(retType)) { (_, c, a, errorID) =>
        val i = genUID()
        ToArray(StreamMap(ToStream(a), i, irOp(c, Ref(i, c.typ), errorID)))
      }

      registerIR2(stringOp, TArray(argType), TArray(argType), TArray(retType)) {
        (_, array1, array2, errorID) =>
          val a1id = genUID()
          val e1 = Ref(a1id, tcoerce[TArray](array1.typ).elementType)
          val a2id = genUID()
          val e2 = Ref(a2id, tcoerce[TArray](array2.typ).elementType)
          ToArray(StreamZip(
            FastSeq(ToStream(array1), ToStream(array2)),
            FastSeq(a1id, a2id),
            irOp(e1, e2, errorID),
            ArrayZipBehavior.AssertSameLength,
          ))
      }
    }

    registerIR1("sum", TArray(tnum("T")), tv("T"))((_, a, _) => sum(a))

    registerIR1("product", TArray(tnum("T")), tv("T"))((_, a, _) => product(a))

    def makeMinMaxOp(op: String): Seq[IR] => IR = { case Seq(a) =>
      val t = tcoerce[TArray](a.typ).elementType
      val value = genUID()
      val first = genUID()
      val acc = genUID()
      StreamFold2(
        ToStream(a),
        FastSeq((acc, NA(t)), (first, True())),
        value,
        FastSeq(
          If(Ref(first, TBoolean), Ref(value, t), invoke(op, t, Ref(acc, t), Ref(value, t))),
          False(),
        ),
        Ref(acc, t),
      )
    }

    registerIR("min", Array(TArray(tnum("T"))), tv("T"), inline = true)((_, a, _) =>
      makeMinMaxOp("min")(a)
    )
    registerIR("nanmin", Array(TArray(tnum("T"))), tv("T"), inline = true)((_, a, _) =>
      makeMinMaxOp("nanmin")(a)
    )
    registerIR("max", Array(TArray(tnum("T"))), tv("T"), inline = true)((_, a, _) =>
      makeMinMaxOp("max")(a)
    )
    registerIR("nanmax", Array(TArray(tnum("T"))), tv("T"), inline = true)((_, a, _) =>
      makeMinMaxOp("nanmax")(a)
    )

    registerIR("mean", Array(TArray(tnum("T"))), TFloat64, inline = true)((_, a, _) => mean(a))

    registerIR1("median", TArray(tnum("T")), tv("T")) { (_, array, errorID) =>
      val t = array.typ.asInstanceOf[TArray].elementType
      val v = Ref(genUID(), t)
      val a = Ref(genUID(), TArray(t))
      val size = Ref(genUID(), TInt32)
      val lastIdx = size - 1
      val midIdx = lastIdx.floorDiv(2)
      def ref(i: IR) = ArrayRef(a, i, errorID)
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      Let(
        FastSeq(a.name -> ArraySort(StreamFilter(ToStream(array), v.name, !IsNA(v)))),
        If(
          IsNA(a),
          NA(t),
          Let(
            FastSeq(size.name -> ArrayLen(a)),
            If(
              size.ceq(0),
              NA(t),
              If(
                invoke("mod", TInt32, size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)),
              ),
            ),
          ),
        ),
      )
    }

    def argF(a: IR, op: (Type) => ComparisonOp[Boolean], errorID: Int): IR = {
      val t = tcoerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32)
      val accum = genUID()
      val value = genUID()
      val m = genUID()
      val idx = genUID()

      def updateAccum(min: IR, midx: IR): IR =
        MakeStruct(FastSeq("m" -> min, "midx" -> midx))

      GetField(
        StreamFold(
          StreamRange(I32(0), ArrayLen(a), I32(1)),
          NA(tAccum),
          accum,
          idx,
          Let(
            FastSeq(
              value -> ArrayRef(a, Ref(idx, TInt32), errorID),
              m -> GetField(Ref(accum, tAccum), "m"),
            ),
            If(
              IsNA(Ref(value, t)),
              Ref(accum, tAccum),
              If(
                IsNA(Ref(m, t)),
                updateAccum(Ref(value, t), Ref(idx, TInt32)),
                If(
                  ApplyComparisonOp(op(t), Ref(value, t), Ref(m, t)),
                  updateAccum(Ref(value, t), Ref(idx, TInt32)),
                  Ref(accum, tAccum),
                ),
              ),
            ),
          ),
        ),
        "midx",
      )
    }

    registerIR1("argmin", TArray(tv("T")), TInt32)((_, a, errorID) => argF(a, LT(_), errorID))

    registerIR1("argmax", TArray(tv("T")), TInt32)((_, a, errorID) => argF(a, GT(_), errorID))

    def uniqueIndex(a: IR, op: (Type) => ComparisonOp[Boolean], errorID: Int): IR = {
      val t = tcoerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32, "count" -> TInt32)
      val accum = genUID()
      val value = genUID()
      val m = genUID()
      val idx = genUID()
      val result = genUID()

      def updateAccum(m: IR, midx: IR, count: IR): IR =
        MakeStruct(FastSeq("m" -> m, "midx" -> midx, "count" -> count))

      val fold =
        StreamFold(
          StreamRange(I32(0), ArrayLen(a), I32(1)),
          NA(tAccum),
          accum,
          idx,
          Let(
            FastSeq(
              value -> ArrayRef(a, Ref(idx, TInt32), errorID),
              m -> GetField(Ref(accum, tAccum), "m"),
            ),
            If(
              IsNA(Ref(value, t)),
              Ref(accum, tAccum),
              If(
                IsNA(Ref(m, t)),
                updateAccum(Ref(value, t), Ref(idx, TInt32), I32(1)),
                If(
                  ApplyComparisonOp(op(t), Ref(value, t), Ref(m, t)),
                  updateAccum(Ref(value, t), Ref(idx, TInt32), I32(1)),
                  If(
                    ApplyComparisonOp(EQ(t), Ref(value, t), Ref(m, t)),
                    updateAccum(
                      Ref(value, t),
                      Ref(idx, TInt32),
                      ApplyBinaryPrimOp(Add(), GetField(Ref(accum, tAccum), "count"), I32(1)),
                    ),
                    Ref(accum, tAccum),
                  ),
                ),
              ),
            ),
          ),
        )

      Let(
        FastSeq(result -> fold),
        If(
          ApplyComparisonOp(EQ(TInt32), GetField(Ref(result, tAccum), "count"), I32(1)),
          GetField(Ref(result, tAccum), "midx"),
          NA(TInt32),
        ),
      )
    }

    registerIR1("uniqueMinIndex", TArray(tv("T")), TInt32)((_, a, errorID) =>
      uniqueIndex(a, LT(_), errorID)
    )

    registerIR1("uniqueMaxIndex", TArray(tv("T")), TInt32)((_, a, errorID) =>
      uniqueIndex(a, GT(_), errorID)
    )

    registerIR2("indexArray", TArray(tv("T")), TInt32, tv("T")) { (_, a, i, errorID) =>
      ArrayRef(
        a,
        If(ApplyComparisonOp(LT(TInt32), i, I32(0)), ApplyBinaryPrimOp(Add(), ArrayLen(a), i), i),
        errorID,
      )
    }

    registerIR1("flatten", TArray(TArray(tv("T"))), TArray(tv("T"))) { (_, a, _) =>
      val elt = Ref(genUID(), tcoerce[TArray](a.typ).elementType)
      ToArray(StreamFlatMap(ToStream(a), elt.name, ToStream(elt)))
    }

    registerSCode4(
      "lowerBound",
      TArray(tv("T")),
      tv("T"),
      TInt32,
      TInt32,
      TInt32,
      (_, _, _, _, _) => SInt32,
    ) { case (_, cb, _, array, key, begin, end, _) =>
      val lt =
        cb.emb.ecb.getOrderingFunction(key.st, array.asIndexable.st.elementType, CodeOrdering.Lt())
      primitive(BinarySearch.lowerBound(
        cb,
        array.asIndexable,
        elt => lt(cb, cb.memoize(elt), EmitValue.present(key)),
        begin.asInt.value,
        end.asInt.value,
      ))
    }

    registerIEmitCode2(
      "corr",
      TArray(TFloat64),
      TArray(TFloat64),
      TFloat64,
      (_: Type, _: EmitType, _: EmitType) => EmitType(SFloat64, false),
    ) { case (cb, _, _, errorID, ec1, ec2) =>
      ec1.toI(cb).flatMap(cb) { case pv1: SIndexableValue =>
        ec2.toI(cb).flatMap(cb) { case pv2: SIndexableValue =>
          val l1 = cb.newLocal("len1", pv1.loadLength())
          val l2 = cb.newLocal("len2", pv2.loadLength())
          cb.if_(
            l1.cne(l2),
            cb._fatalWithError(
              errorID,
              "'corr': cannot compute correlation between arrays of different lengths: ",
              l1.toS,
              ", ",
              l2.toS,
            ),
          )
          IEmitCode(
            cb,
            l1.ceq(0), {
              val xSum = cb.newLocal[Double]("xSum", 0d)
              val ySum = cb.newLocal[Double]("ySum", 0d)
              val xSqSum = cb.newLocal[Double]("xSqSum", 0d)
              val ySqSum = cb.newLocal[Double]("ySqSum", 0d)
              val xySum = cb.newLocal[Double]("xySum", 0d)
              val i = cb.newLocal[Int]("i")
              val n = cb.newLocal[Int]("n", 0)
              cb.for_(
                cb.assign(i, 0),
                i < l1,
                cb.assign(i, i + 1), {
                  pv1.loadElement(cb, i).consume(
                    cb,
                    {},
                    { xc =>
                      pv2.loadElement(cb, i).consume(
                        cb,
                        {},
                        { yc =>
                          val x = cb.newLocal[Double]("x", xc.asDouble.value)
                          val y = cb.newLocal[Double]("y", yc.asDouble.value)
                          cb.assign(xSum, xSum + x)
                          cb.assign(xSqSum, xSqSum + x * x)
                          cb.assign(ySum, ySum + y)
                          cb.assign(ySqSum, ySqSum + y * y)
                          cb.assign(xySum, xySum + x * y)
                          cb.assign(n, n + 1)
                        },
                      )
                    },
                  )
                },
              )
              val res =
                cb.memoize((n.toD * xySum - xSum * ySum) / Code.invokeScalaObject1[Double, Double](
                  MathFunctions.mathPackageClass,
                  "sqrt",
                  (n.toD * xSqSum - xSum * xSum) * (n.toD * ySqSum - ySum * ySum),
                ))
              primitive(res)
            },
          )
        }
      }
    }

    registerIEmitCode4(
      "local_to_global_g",
      TArray(TVariable("T")),
      TArray(TInt32),
      TInt32,
      TVariable("T"),
      TArray(TVariable("T")),
      { case (_, inArrayET, la, n, _) =>
        EmitType(
          PCanonicalArray(
            PType.canonical(inArrayET.st.asInstanceOf[SContainer].elementType.storageType())
          ).sType,
          inArrayET.required && la.required && n.required,
        )
      },
    ) {
      case (
            cb,
            region,
            rt: SIndexablePointer,
            err,
            array,
            localAlleles,
            nTotalAlleles,
            fillInValue,
          ) =>
        IEmitCode.multiMapEmitCodes(cb, FastSeq(array, localAlleles, nTotalAlleles)) {
          case IndexedSeq(
                array: SIndexableValue,
                localAlleles: SIndexableValue,
                _nTotalAlleles: SInt32Value,
              ) =>
            def triangle(x: Value[Int]): Code[Int] = (x * (x + 1)) / 2
            val nTotalAlleles = _nTotalAlleles.value
            val nGenotypes = cb.memoize(triangle(nTotalAlleles))
            val pt = rt.pType.asInstanceOf[PCanonicalArray]
            cb.if_(
              nTotalAlleles < 0,
              cb._fatalWithError(
                err,
                "local_to_global: n_total_alleles less than 0: ",
                nGenotypes.toS,
              ),
            )
            val localLen = array.loadLength()
            val laLen = localAlleles.loadLength()
            cb.if_(
              localLen cne triangle(laLen),
              cb._fatalWithError(
                err,
                "local_to_global: array should be the triangle number of local alleles: found: ",
                localLen.toS,
                " elements, and",
                laLen.toS,
                " alleles",
              ),
            )

            val fillIn = cb.memoize(fillInValue)

            val (push, finish) = pt.constructFromIndicesUnsafe(cb, region, nGenotypes, false)

            // fill in if necessary
            cb.if_(
              localLen cne nGenotypes, {
                val i = cb.newLocal[Int]("i", 0)
                cb.while_(
                  i < nGenotypes, {
                    push(cb, i, fillIn.toI(cb))
                    cb.assign(i, i + 1)
                  },
                )
              },
            )

            val i = cb.newLocal[Int]("la_i", 0)
            val laGIndexer = cb.newLocal[Int]("g_indexer", 0)
            cb.while_(
              i < laLen, {
                val lai = localAlleles.loadElement(cb, i).get(
                  cb,
                  "local_to_global: local alleles elements cannot be missing",
                  err,
                ).asInt32.value
                cb.if_(
                  lai >= nTotalAlleles,
                  cb._fatalWithError(
                    err,
                    "local_to_global: local allele of ",
                    lai.toS,
                    " out of bounds given n_total_alleles of ",
                    nTotalAlleles.toS,
                  ),
                )

                val j = cb.newLocal[Int]("la_j", 0)
                cb.while_(
                  j <= i, {
                    val laj = localAlleles.loadElement(cb, j).get(
                      cb,
                      "local_to_global: local alleles elements cannot be missing",
                      err,
                    ).asInt32.value

                    val dest = cb.newLocal[Int]("dest")
                    cb.if_(
                      lai >= laj,
                      cb.assign(dest, triangle(lai) + laj),
                      cb.assign(dest, triangle(laj) + lai),
                    )

                    push(cb, dest, array.loadElement(cb, laGIndexer))
                    cb.assign(laGIndexer, laGIndexer + 1)
                    cb.assign(j, j + 1)
                  },
                )

                cb.assign(i, i + 1)
              },
            )

            finish(cb)
        }
    }

    registerIEmitCode5(
      "local_to_global_a_r",
      TArray(TVariable("T")),
      TArray(TInt32),
      TInt32,
      TVariable("T"),
      TBoolean,
      TArray(TVariable("T")),
      { case (_, inArrayET, la, n, _, omitFirst) =>
        EmitType(
          PCanonicalArray(
            PType.canonical(inArrayET.st.asInstanceOf[SContainer].elementType.storageType())
          ).sType,
          inArrayET.required && la.required && n.required && omitFirst.required,
        )
      },
    ) {
      case (
            cb,
            region,
            rt: SIndexablePointer,
            err,
            array,
            localAlleles,
            nTotalAlleles,
            fillInValue,
            omitFirstElement,
          ) =>
        IEmitCode.multiMapEmitCodes(
          cb,
          FastSeq(array, localAlleles, nTotalAlleles, omitFirstElement),
        ) {
          case IndexedSeq(
                array: SIndexableValue,
                localAlleles: SIndexableValue,
                _nTotalAlleles: SInt32Value,
                omitFirst: SBooleanValue,
              ) =>
            val nTotalAlleles = _nTotalAlleles.value
            val pt = rt.pType.asInstanceOf[PCanonicalArray]
            cb.if_(
              nTotalAlleles < 0,
              cb._fatalWithError(
                err,
                "local_to_global: n_total_alleles less than 0: ",
                nTotalAlleles.toS,
              ),
            )
            val localLen = array.loadLength()
            cb.if_(
              localLen cne localAlleles.loadLength(),
              cb._fatalWithError(
                err,
                "local_to_global: array and local alleles lengths differ: ",
                localLen.toS,
                ", ",
                localAlleles.loadLength().toS,
              ),
            )

            val fillIn = cb.memoize(fillInValue)

            val idxAdjustmentForOmitFirst = cb.newLocal[Int]("idxAdj")
            cb.if_(
              omitFirst.value,
              cb.assign(idxAdjustmentForOmitFirst, 1),
              cb.assign(idxAdjustmentForOmitFirst, 0),
            )

            val globalLen = cb.memoize(nTotalAlleles - idxAdjustmentForOmitFirst)

            val (push, finish) = pt.constructFromIndicesUnsafe(cb, region, globalLen, false)

            // fill in if necessary
            cb.if_(
              localLen cne globalLen, {
                val i = cb.newLocal[Int]("i", 0)
                cb.while_(
                  i < globalLen, {
                    push(cb, i, fillIn.toI(cb))
                    cb.assign(i, i + 1)
                  },
                )
              },
            )

            val i = cb.newLocal[Int]("la_i", 0)
            cb.while_(
              i < localLen, {
                val lai = localAlleles.loadElement(cb, i + idxAdjustmentForOmitFirst).get(
                  cb,
                  "local_to_global: local alleles elements cannot be missing",
                  err,
                ).asInt32.value
                cb.if_(
                  lai >= nTotalAlleles,
                  cb._fatalWithError(
                    err,
                    "local_to_global: local allele of ",
                    lai.toS,
                    " out of bounds given n_total_alleles of ",
                    nTotalAlleles.toS,
                  ),
                )
                push(cb, cb.memoize(lai - idxAdjustmentForOmitFirst), array.loadElement(cb, i))

                cb.assign(i, i + 1)
              },
            )

            finish(cb)
        }
    }
  }
}

package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
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
          Apply("pow", Seq(), FastSeq(ir1, ir2), TFloat64, errorID),
      ),
      (
        "mod",
        tnum("T"),
        tv("T"),
        (ir1: IR, ir2: IR, errorID: Int) =>
          Apply("mod", Seq(), FastSeq(ir1, ir2), ir2.typ, errorID),
      ),
    )

  def mean(args: Seq[IR]): IR = {
    val Seq(a) = args
    val t = tcoerce[TArray](a.typ).elementType
    val elt = freshName()
    val n = freshName()
    val sum = freshName()
    StreamFold2(
      ToStream(a),
      FastSeq((n, I32(0)), (sum, zero(t))),
      elt,
      FastSeq(Ref(n, TInt32) + I32(1), Ref(sum, t) + Ref(elt, t)),
      Cast(Ref(sum, t), TFloat64) / Cast(Ref(n, TInt32), TFloat64),
    )
  }

  def isEmpty(a: IR): IR = ApplyComparisonOp(EQ, ArrayLen(a), I32(0))

  def extend(a1: IR, a2: IR): IR = {
    val uid = freshName()
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

  def exists(a: IR, cond: IR => IR): IR = foldIR(ToStream(a), False()) { (acc, elt) =>
    invoke("lor", TBoolean, acc, cond(elt))
  }

  def contains(a: IR, value: IR): IR =
    exists(a, elt => ApplyComparisonOp(EQWithNA, elt, value))

  def sum(a: IR): IR = {
    val t = tcoerce[TArray](a.typ).elementType
    val zero = Cast(I64(0), t)
    foldIR(ToStream(a), zero)((sum, v) => ApplyBinaryPrimOp(Add(), sum, v))
  }

  def product(a: IR): IR = {
    val t = tcoerce[TArray](a.typ).elementType
    val one = Cast(I64(1), t)
    foldIR(ToStream(a), one)((product, v) => ApplyBinaryPrimOp(Multiply(), product, v))
  }

  override def registerAll(): Unit = {
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
        mapArray(a)(i => irOp(i, c, errorID))
      }

      registerIR2(stringOp, argType, TArray(argType), TArray(retType)) { (_, c, a, errorID) =>
        mapArray(a)(i => irOp(c, i, errorID))
      }

      registerIR2(stringOp, TArray(argType), TArray(argType), TArray(retType)) {
        (_, array1, array2, errorID) =>
          ToArray(zipIR(
            FastSeq(ToStream(array1), ToStream(array2)),
            ArrayZipBehavior.AssertSameLength,
          ) { case Seq(a1id, a2id) =>
            irOp(a1id, a2id, errorID)
          })
      }
    }

    registerIR1("sum", TArray(tnum("T")), tv("T"))((_, a, _) => sum(a))

    registerIR1("product", TArray(tnum("T")), tv("T"))((_, a, _) => product(a))

    def makeMinMaxOp(op: String): Seq[IR] => IR = { case Seq(a) =>
      val t = tcoerce[TArray](a.typ).elementType
      val value = freshName()
      val first = freshName()
      val acc = freshName()
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

      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      bindIR(ArraySort(filterIR(ToStream(array))(!IsNA(_)))) { a =>
        def ref(i: IR) = ArrayRef(a, i, errorID)
        If(
          IsNA(a),
          NA(t),
          bindIR(ArrayLen(a)) { size =>
            val lastIdx = size - 1
            val midIdx = lastIdx.floorDiv(2)
            If(
              size.ceq(0),
              NA(t),
              If(
                invoke("mod", TInt32, size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)),
              ),
            )
          },
        )
      }
    }

    def argF(a: IR, op: ComparisonOp[Boolean], errorID: Int): IR = {
      val t = tcoerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32)

      def updateAccum(min: IR, midx: IR): IR =
        MakeStruct(FastSeq("m" -> min, "midx" -> midx))

      GetField(
        foldIR(StreamRange(I32(0), ArrayLen(a), I32(1)), NA(tAccum)) { (accum, idx) =>
          bindIRs(ArrayRef(a, idx, errorID), GetField(accum, "m")) { case Seq(value, m) =>
            If(
              IsNA(value),
              accum,
              If(
                IsNA(m),
                updateAccum(value, idx),
                If(
                  ApplyComparisonOp(op, value, m),
                  updateAccum(value, idx),
                  accum,
                ),
              ),
            )
          }
        },
        "midx",
      )
    }

    registerIR1("argmin", TArray(tv("T")), TInt32)((_, a, errorID) => argF(a, LT, errorID))

    registerIR1("argmax", TArray(tv("T")), TInt32)((_, a, errorID) => argF(a, GT, errorID))

    def uniqueIndex(a: IR, op: ComparisonOp[Boolean], errorID: Int): IR = {
      val t = tcoerce[TArray](a.typ).elementType
      val tAccum = TStruct("m" -> t, "midx" -> TInt32, "count" -> TInt32)

      def updateAccum(m: IR, midx: IR, count: IR): IR =
        MakeStruct(FastSeq("m" -> m, "midx" -> midx, "count" -> count))

      val fold = foldIR(StreamRange(I32(0), ArrayLen(a), I32(1)), NA(tAccum)) { (accum, idx) =>
        bindIRs(ArrayRef(a, idx, errorID), GetField(accum, "m")) { case Seq(value, m) =>
          If(
            IsNA(value),
            accum,
            If(
              IsNA(m),
              updateAccum(value, idx, I32(1)),
              If(
                ApplyComparisonOp(op, value, m),
                updateAccum(value, idx, I32(1)),
                If(
                  ApplyComparisonOp(EQ, value, m),
                  updateAccum(
                    value,
                    idx,
                    ApplyBinaryPrimOp(Add(), GetField(accum, "count"), I32(1)),
                  ),
                  accum,
                ),
              ),
            ),
          )
        }
      }

      bindIR(fold) { result =>
        If(
          ApplyComparisonOp(EQ, GetField(result, "count"), I32(1)),
          GetField(result, "midx"),
          NA(TInt32),
        )
      }
    }

    registerIR1("uniqueMinIndex", TArray(tv("T")), TInt32)((_, a, errorID) =>
      uniqueIndex(a, LT, errorID)
    )

    registerIR1("uniqueMaxIndex", TArray(tv("T")), TInt32)((_, a, errorID) =>
      uniqueIndex(a, GT, errorID)
    )

    registerIR2("indexArray", TArray(tv("T")), TInt32, tv("T")) { (_, a, i, errorID) =>
      ArrayRef(
        a,
        If(ApplyComparisonOp(LT, i, I32(0)), ApplyBinaryPrimOp(Add(), ArrayLen(a), i), i),
        errorID,
      )
    }

    registerIR1("flatten", TArray(TArray(tv("T"))), TArray(tv("T"))) { (_, a, _) =>
      ToArray(flatMapIR(ToStream(a))(ToStream(_)))
    }

    /* Construct an array of length `len`, with the values in `elts` copied to the positions in
     * `indices` */
    registerSCode3t(
      "scatter",
      Array(tv("T")),
      TArray(tv("T")), // elts
      TArray(TInt32), // indices
      TInt32, // len
      TArray(tv("T")),
      (_, a, _, _) => PCanonicalArray(a.asInstanceOf[SContainer].elementType.storageType()).sType,
    ) {
      case (
            er,
            cb,
            _,
            rt: SIndexablePointer,
            elts: SIndexableValue,
            indices: SIndexableValue,
            len: SInt32Value,
            errorID,
          ) =>
        cb.if_(
          elts.loadLength.cne(indices.loadLength),
          cb._fatalWithError(errorID, "scatter: values and indices arrays have different lengths"),
        )
        cb.if_(
          elts.loadLength > len.value,
          cb._fatalWithError(errorID, "scatter: values array is larger than result length"),
        )
        val pt = rt.pType.asInstanceOf[PCanonicalArray]
        val (push, finish) =
          pt.constructFromIndicesUnsafe(cb, er, len.value, deepCopy = false)
        indices.forEachDefined(cb) { case (cb, pos, idx: SInt32Value) =>
          cb.if_(
            idx.value < 0 || idx.value >= len.value,
            cb._fatalWithError(
              errorID,
              "scatter: indices array contained index ",
              idx.value.toS,
              ", which is greater than result length ",
              len.value.toS,
            ),
          )
          push(cb, idx.value, elts.loadElement(cb, pos))
        }
        finish(cb)
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
          val l1 = cb.newLocal("len1", pv1.loadLength)
          val l2 = cb.newLocal("len2", pv2.loadLength)
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
            val localLen = array.loadLength
            val laLen = localAlleles.loadLength
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
                val lai = localAlleles.loadElement(cb, i).getOrFatal(
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
                    val laj = localAlleles.loadElement(cb, j).getOrFatal(
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
            val localLen = array.loadLength
            cb.if_(
              localLen cne localAlleles.loadLength,
              cb._fatalWithError(
                err,
                "local_to_global: array and local alleles lengths differ: ",
                localLen.toS,
                ", ",
                localAlleles.loadLength.toS,
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
                val lai = localAlleles.loadElement(cb, i + idxAdjustmentForOmitFirst).getOrFatal(
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

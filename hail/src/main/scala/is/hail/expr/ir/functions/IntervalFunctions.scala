package is.hail.expr.ir.functions

import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SIntervalPointer
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SBooleanValue}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq

object IntervalFunctions extends RegistryFunctions {

  def pointLTIntervalEndpoint(cb: EmitCodeBuilder,
    point: SValue, endpoint: SValue, leansRight: Code[Boolean]
  ): Code[Boolean] = {
    val ord = cb.emb.ecb.getOrdering(point.st, endpoint.st)
    val result = cb.newLocal[Int]("comparePointWithIntervalEndpoint")

    cb.assign(result, ord.compareNonnull(cb, point, endpoint))
    (result < 0) || (result.ceq(0) && leansRight)
  }

  def pointGTIntervalEndpoint(cb: EmitCodeBuilder,
    point: SValue, endpoint: SValue, leansRight: Code[Boolean]
  ): Code[Boolean] = {
    val ord = cb.emb.ecb.getOrdering(point.st, endpoint.st)
    val result = cb.newLocal[Int]("comparePointWithIntervalEndpoint")

    cb.assign(result, ord.compareNonnull(cb, point, endpoint))
    (result > 0) || (result.ceq(0) && !leansRight)
  }

  def intervalEndpointCompare(cb: EmitCodeBuilder,
    lhs: SValue, lhsLeansRight: Code[Boolean],
    rhs: SValue, rhsLeansRight: Code[Boolean]
  ): Value[Int] = {
    val ord = cb.emb.ecb.getOrdering(lhs.st, rhs.st)
    val result = cb.newLocal[Int]("intervalEndpointCompare")

    cb.assign(result, ord.compareNonnull(cb, lhs, rhs))
    cb.ifx(result.ceq(0),
      cb.assign(result, lhsLeansRight.toI - rhsLeansRight.toI))
    result
  }

  def pointIntervalCompare(cb: EmitCodeBuilder, point: SValue, interval: SIntervalValue): IEmitCode = {
    interval.loadStart(cb).flatMap(cb) { start =>
      cb.ifx(pointLTIntervalEndpoint(cb, point, start, !interval.includesStart()), {
        IEmitCode.present(cb, primitive(const(-1)))
      }, {
        interval.loadEnd(cb).map(cb) { end =>
          cb.ifx(pointLTIntervalEndpoint(cb, point, end, interval.includesEnd()), {
            primitive(const(0))
          }, {
            primitive(const(1))
          })
        }
      })
    }
  }

  def intervalContains(cb: EmitCodeBuilder, interval: SIntervalValue, point: SValue): IEmitCode = {
    interval.loadStart(cb).flatMap(cb) { start =>
      cb.ifx(pointGTIntervalEndpoint(cb, point, start, !interval.includesStart()),
        interval.loadEnd(cb).map(cb) { end =>
          primitive(cb.memoize(pointLTIntervalEndpoint(cb, point, end, interval.includesEnd())))
        },
        IEmitCode.present(cb, primitive(false)))
    }
  }

  def intervalsOverlap(cb: EmitCodeBuilder, lhs: SIntervalValue, rhs: SIntervalValue): IEmitCode = {
    IEmitCode.multiFlatMap(cb,
      FastIndexedSeq(lhs.loadEnd, rhs.loadStart)
    ) { case Seq(lEnd, rStart) =>
      cb.ifx(intervalEndpointCompare(cb, lEnd, lhs.includesEnd(), rStart, !rhs.includesStart()) > 0, {
        IEmitCode.multiMap(cb,
          FastIndexedSeq(lhs.loadStart, rhs.loadEnd)
        ) { case Seq(lStart, rEnd) =>
          primitive(cb.memoize(intervalEndpointCompare(cb, rEnd, rhs.includesEnd(), lStart, !lhs.includesStart()) > 0))
        }
      }, {
        IEmitCode.present(cb, primitive(const(false)))
      })
    }
  }

  def compareStructWithIntervalEndpoint(cb: EmitCodeBuilder, point: SBaseStructValue, intervalEndpoint: SBaseStructValue): Value[Int] = {
    val endpoint = intervalEndpoint.loadField(cb, 0).get(cb).asBaseStruct
    val endpointLength = intervalEndpoint.loadField(cb, 1).get(cb).asInt.value

    val c = cb.newLocal[Int]("c", 0)
    (0 until endpoint.st.size).foreach { idx =>
      cb.ifx(c.ceq(0) && const(idx) < endpointLength, {
        val endpointField = cb.memoize(endpoint.loadField(cb, idx))
        val pointField = cb.memoize(point.loadField(cb, idx))
        cb.assign(c,
          cb.emb.ecb.getOrderingFunction(pointField.st, endpointField.st, CodeOrdering.Compare())
            .apply(cb, pointField, endpointField))
      })
    }
    c
  }

  def registerAll(): Unit = {
    registerIEmitCode4("Interval", tv("T"), tv("T"), TBoolean, TBoolean, TInterval(tv("T")),
      { case (_: Type, startpt, endpt, includesStartET, includesEndET) =>
        EmitType(PCanonicalInterval(
          InferPType.getCompatiblePType(Seq(startpt.typeWithRequiredness.canonicalPType, endpt.typeWithRequiredness.canonicalPType)),
          required = includesStartET.required && includesEndET.required
        ).sType, includesStartET.required && includesEndET.required)
      }) {
      case (cb, r, SIntervalPointer(pt: PCanonicalInterval), _, start, end, includesStart, includesEnd) =>

        includesStart.toI(cb).flatMap(cb) { includesStart =>
          includesEnd.toI(cb).map(cb) { includesEnd =>

            pt.constructFromCodes(cb, r,
              start,
              end,
              includesStart.asBoolean.value,
              includesEnd.asBoolean.value)
          }
        }
    }

    registerIEmitCode1("start", TInterval(tv("T")), tv("T"),
      (_: Type, x: EmitType) => EmitType(x.st.asInstanceOf[SInterval].pointType, x.required && x.st.asInstanceOf[SInterval].pointEmitType.required)) {
      case (cb, r, rt, _, interval) =>
        interval.toI(cb).flatMap(cb) { case pv: SIntervalValue =>
          pv.loadStart(cb)
        }
    }

    registerIEmitCode1("end", TInterval(tv("T")), tv("T"),
      (_: Type, x: EmitType) => EmitType(x.st.asInstanceOf[SInterval].pointType, x.required && x.st.asInstanceOf[SInterval].pointEmitType.required)) {
      case (cb, r, rt, _, interval) =>
        interval.toI(cb).flatMap(cb) { case pv: SIntervalValue =>
          pv.loadEnd(cb)
        }
    }

    registerSCode1("includesStart", TInterval(tv("T")), TBoolean, (_: Type, x: SType) =>
      SBoolean
    ) {
      case (r, cb, rt, interval: SIntervalValue, _) => primitive(interval.includesStart())
    }

    registerSCode1("includesEnd", TInterval(tv("T")), TBoolean, (_: Type, x: SType) =>
      SBoolean
    ) {
      case (r, cb, rt, interval: SIntervalValue, _) => primitive(interval.includesEnd())
    }

    registerIEmitCode2("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(_: Type, intervalT: EmitType, pointT: EmitType) =>
        val intervalST = intervalT.st.asInstanceOf[SInterval]
        val required = intervalT.required && intervalST.pointEmitType.required && pointT.required
        EmitType(SBoolean, required)
    }) { case (cb, r, rt, _, int, point) =>
      IEmitCode.multiFlatMap(cb,
        FastIndexedSeq(int.toI, point.toI)
      ) { case Seq(interval: SIntervalValue, point) =>
        intervalContains(cb, interval, point)
      }
    }

    registerSCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: SType) => SBoolean) {
      case (r, cb, rt, interval: SIntervalValue, _) =>
        primitive(interval.isEmpty(cb))
    }

    registerIEmitCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, {
      (_: Type, i1t: EmitType, i2t: EmitType) =>
        val i1ST = i1t.st.asInstanceOf[SInterval]
        val i2ST = i2t.st.asInstanceOf[SInterval]
        val required = i1t.required && i2t.required && i1ST.pointEmitType.required && i2ST.pointEmitType.required
        EmitType(SBoolean, required)
    }) { case (cb, r, rt, _, interval1: EmitCode, interval2: EmitCode) =>
      IEmitCode.multiFlatMap(cb, FastIndexedSeq(interval1.toI, interval2.toI)) {
        case Seq(interval1: SIntervalValue, interval2: SIntervalValue) =>
        intervalsOverlap(cb, interval1, interval2)
      }
    }

    registerIR2("sortedNonOverlappingIntervalsContain",
      TArray(TInterval(tv("T"))), tv("T"), TBoolean) { case (_, intervals, value, errorID) =>
      val uid = genUID()
      val uid2 = genUID()
      Let(uid, LowerBoundOnOrderedCollection(intervals, value, onKey = true),
        (Let(uid2, Ref(uid, TInt32) - I32(1), (Ref(uid2, TInt32) >= 0)
          && invoke("contains", TBoolean, ArrayRef(intervals, Ref(uid2, TInt32), errorID), value)))
          || ((Ref(uid, TInt32) < ArrayLen(intervals))
          && invoke("contains", TBoolean, ArrayRef(intervals, Ref(uid, TInt32), errorID), value)))
    }

    val endpointT = TTuple(tv("T"), TInt32)
    registerSCode3("pointLessThanPartitionIntervalLeftEndpoint", tv("T"), endpointT, TBoolean, TBoolean, (_, _, _, _) => SBoolean) {
      case (_, cb, _, point: SBaseStructValue, leftPartitionEndpoint: SBaseStructValue, containsStart: SBooleanValue, _) =>
        val c = compareStructWithIntervalEndpoint(cb, point, leftPartitionEndpoint)
        primitive(cb.memoize((c < 0) || (c.ceq(0) && !containsStart.value)))
    }

    registerSCode3("pointLessThanPartitionIntervalRightEndpoint", tv("T"), endpointT, TBoolean, TBoolean, (_, _, _, _) => SBoolean) {
      case (_, cb, _, point: SBaseStructValue, rightPartitionEndpoint: SBaseStructValue, containsEnd: SBooleanValue, _) =>
        val c = compareStructWithIntervalEndpoint(cb, point, rightPartitionEndpoint)
        primitive(cb.memoize((c < 0) || (c.ceq(0) && containsEnd.value)))
    }

    registerSCode2("partitionIntervalContains",
      TStruct("left" -> endpointT, "right" -> endpointT, "includesLeft" -> TBoolean, "includesRight" -> TBoolean),
      tv("T"), TBoolean, (_, _, _) => SBoolean) {
      case (_, cb, _, interval: SBaseStructValue, point: SBaseStructValue, _) =>
        val leftTuple = interval.loadField(cb, "left").get(cb).asBaseStruct

        val c = compareStructWithIntervalEndpoint(cb, point, leftTuple)

        val includesLeft = interval.loadField(cb, "includesLeft").get(cb).asBoolean.value
        val pointGTLeft = (c > 0) || (c.ceq(0) && includesLeft)

        val isContained = cb.newLocal[Boolean]("partitionInterval_b", pointGTLeft)

        cb.ifx(isContained, {
          // check right endpoint
          val rightTuple = interval.loadField(cb, "right").get(cb).asBaseStruct

          val c = compareStructWithIntervalEndpoint(cb, point, rightTuple)
          val includesRight = interval.loadField(cb, "includesRight").get(cb).asBoolean.value
          val pointLTRight = (c < 0) || (c.ceq(0) && includesRight)
          cb.assign(isContained, pointLTRight)
        })

        primitive(isContained)
    }
  }
}

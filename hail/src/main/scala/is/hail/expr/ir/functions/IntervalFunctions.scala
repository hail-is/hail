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

object IntervalFunctions extends RegistryFunctions {

  def pointLTIntervalEndpoint(cb: EmitCodeBuilder,
    point: EmitValue, endpoint: EmitValue, leansRight: Code[Boolean], missingEqual: Boolean = true
  ): Code[Boolean] = {
    val compare = cb.emb.ecb.getOrderingFunction(point.st, endpoint.st, CodeOrdering.Compare(missingEqual))

    val result = cb.newLocal[Int]("comparePointWithIntervalEndpoint")
    cb.assign(result, compare(cb, point, endpoint))
    (result < 0) || (result.ceq(0) && leansRight)
  }

  def pointGTIntervalEndpoint(cb: EmitCodeBuilder,
    point: EmitValue, endpoint: EmitValue, leansRight: Code[Boolean], missingEqual: Boolean = true
  ): Code[Boolean] = {
    val compare = cb.emb.ecb.getOrderingFunction(point.st, endpoint.st, CodeOrdering.Compare(missingEqual))

    val result = cb.newLocal[Int]("comparePointWithIntervalEndpoint")
    cb.assign(result, compare(cb, point, endpoint))
    (result > 0) || (result.ceq(0) && !leansRight)
  }

  def intervalEndpointCompare(cb: EmitCodeBuilder,
    lhs: EmitValue, lhsLeansRight: Code[Boolean],
    rhs: EmitValue, rhsLeansRight: Code[Boolean],
    missingEqual: Boolean = true
  ): Value[Int] = {
    val compare = cb.emb.ecb.getOrderingFunction(lhs.st, rhs.st, CodeOrdering.Compare(missingEqual))

    val result = cb.newLocal[Int]("intervalEndpointCompare")
    cb.assign(result, compare(cb, lhs, rhs))
    cb.ifx(result.ceq(0),
      cb.assign(result, lhsLeansRight.toI - rhsLeansRight.toI))
    result
  }

  def pointIntervalCompare(cb: EmitCodeBuilder, point: EmitValue, interval: SIntervalValue, missingEqual: Boolean = true): Value[Int] = {
    val result = cb.newLocal[Int]("pointIntervalCompare")
    val start = cb.memoize(interval.loadStart(cb))
    cb.ifx(pointLTIntervalEndpoint(cb, point, start, !interval.includesStart(), missingEqual), {
      cb.assign(result, -1)
    }, {
      val end = cb.memoize(interval.loadEnd(cb))
      cb.ifx(pointLTIntervalEndpoint(cb, point, end, interval.includesEnd(), missingEqual), {
        cb.assign(result, 0)
      }, {
        cb.assign(result, 1)
      })
    })
    result
  }

  def intervalContains(cb: EmitCodeBuilder, interval: SIntervalValue, point: EmitValue): Value[Boolean] = {
    val start = cb.memoize(interval.loadStart(cb))
    val contains = cb.newLocal[Boolean]("contains", false)
    cb.ifx(pointGTIntervalEndpoint(cb, point, start, !interval.includesStart()), {
      val end = cb.memoize(interval.loadEnd(cb))
      cb.assign(contains, pointLTIntervalEndpoint(cb, point, end, interval.includesEnd()))
    })
    contains
  }

  def intervalsOverlap(cb: EmitCodeBuilder, lhs: SIntervalValue, rhs: SIntervalValue): Value[Boolean] = {
    val lEnd = cb.memoize(lhs.loadEnd(cb))
    val rStart = cb.memoize(rhs.loadStart(cb))
    val overlaps = cb.newLocal[Boolean]("contains", false)
    cb.ifx(intervalEndpointCompare(cb, lEnd, lhs.includesEnd(), rStart, !rhs.includesStart()) >= 0, {
      val lStart = cb.memoize(lhs.loadStart(cb))
      val rEnd = cb.memoize(rhs.loadEnd(cb))
      cb.assign(overlaps,
        intervalEndpointCompare(cb, rEnd, rhs.includesEnd(), lStart, !lhs.includesStart()) >= 0)
    })
    overlaps
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
          cb.emb.ecb.getOrderingFunction(endpointField.st, pointField.st, CodeOrdering.Compare())
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
      case(_: Type, intervalT: EmitType, _: EmitType) => EmitType(SBoolean, intervalT.required)
    }) {
      case (cb, r, rt, _, int, point) =>
        int.toI(cb).map(cb) { case interval: SIntervalValue =>
          val pointv = cb.memoize(point, "point")
          primitive(intervalContains(cb, interval, pointv))
        }
    }

    registerSCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: SType) => SBoolean) {
      case (r, cb, rt, interval: SIntervalValue, _) =>
        primitive(interval.isEmpty(cb))
    }

    registerSCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, (_: Type, i1t: SType, i2t: SType) => SBoolean) {
      case (r, cb, rt, interval1: SIntervalValue, interval2: SIntervalValue, _) =>
        primitive(intervalsOverlap(cb, interval1, interval2))
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

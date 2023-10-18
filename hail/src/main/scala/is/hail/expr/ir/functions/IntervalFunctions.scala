package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SIntervalPointer, SStackStruct, SStackStructValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SBooleanValue, SInt32}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.virtual._
import is.hail.utils.FastSeq

object IntervalFunctions extends RegistryFunctions {

  def pointLTIntervalEndpoint(cb: EmitCodeBuilder,
    point: SValue, endpoint: SValue, leansRight: Code[Boolean]
  ): Code[Boolean] = {
    val ord = cb.emb.ecb.getOrdering(point.st, endpoint.st)
    val result = ord.compareNonnull(cb, point, endpoint)
    (result < 0) || (result.ceq(0) && leansRight)
  }

  def pointGTIntervalEndpoint(cb: EmitCodeBuilder,
    point: SValue, endpoint: SValue, leansRight: Code[Boolean]
  ): Code[Boolean] = {
    val ord = cb.emb.ecb.getOrdering(point.st, endpoint.st)
    val result = ord.compareNonnull(cb, point, endpoint)
    (result > 0) || (result.ceq(0) && !leansRight)
  }

  def intervalEndpointCompare(cb: EmitCodeBuilder,
    lhs: SValue, lhsLeansRight: Code[Boolean],
    rhs: SValue, rhsLeansRight: Code[Boolean]
  ): Value[Int] = {
    val ord = cb.emb.ecb.getOrdering(lhs.st, rhs.st)
    val result = cb.newLocal[Int]("intervalEndpointCompare")

    cb.assign(result, ord.compareNonnull(cb, lhs, rhs))
    cb.if_(result.ceq(0),
      cb.assign(result, lhsLeansRight.toI - rhsLeansRight.toI))
    result
  }

  def pointIntervalCompare(cb: EmitCodeBuilder, point: SValue, interval: SIntervalValue): IEmitCode = {
    interval.loadStart(cb).flatMap(cb) { start =>
      cb.if_(pointLTIntervalEndpoint(cb, point, start, !interval.includesStart), {
        IEmitCode.present(cb, primitive(const(-1)))
      }, {
        interval.loadEnd(cb).map(cb) { end =>
          cb.if_(pointLTIntervalEndpoint(cb, point, end, interval.includesEnd), {
            primitive(const(0))
          }, {
            primitive(const(1))
          })
        }
      })
    }
  }

  def intervalPointCompare(cb: EmitCodeBuilder, interval: SIntervalValue, point: SValue): IEmitCode = {
    interval.loadStart(cb).flatMap(cb) { start =>
      cb.if_(pointLTIntervalEndpoint(cb, point, start, !interval.includesStart), {
        IEmitCode.present(cb, primitive(const(1)))
      }, {
        interval.loadEnd(cb).map(cb) { end =>
          cb.if_(pointLTIntervalEndpoint(cb, point, end, interval.includesEnd), {
            primitive(const(0))
          }, {
            primitive(const(-1))
          })
        }
      })
    }
  }

  def intervalContains(cb: EmitCodeBuilder, interval: SIntervalValue, point: SValue): IEmitCode = {
    interval.loadStart(cb).flatMap(cb) { start =>
      cb.if_(pointGTIntervalEndpoint(cb, point, start, !interval.includesStart),
        interval.loadEnd(cb).map(cb) { end =>
          primitive(cb.memoize(pointLTIntervalEndpoint(cb, point, end, interval.includesEnd)))
        },
        IEmitCode.present(cb, primitive(false)))
    }
  }

  def intervalsOverlap(cb: EmitCodeBuilder, lhs: SIntervalValue, rhs: SIntervalValue): IEmitCode = {
    IEmitCode.multiFlatMap(cb,
      FastSeq(lhs.loadEnd, rhs.loadStart)
    ) { case Seq(lEnd, rStart) =>
      cb.if_(intervalEndpointCompare(cb, lEnd, lhs.includesEnd, rStart, !rhs.includesStart) > 0, {
        IEmitCode.multiMap(cb,
          FastSeq(lhs.loadStart, rhs.loadEnd)
        ) { case Seq(lStart, rEnd) =>
          primitive(cb.memoize(intervalEndpointCompare(cb, rEnd, rhs.includesEnd, lStart, !lhs.includesStart) > 0))
        }
      }, {
        IEmitCode.present(cb, primitive(const(false)))
      })
    }
  }

  def _partitionIntervalEndpointCompare(cb: EmitCodeBuilder,
    lStruct: SBaseStructValue, lLength: Value[Int], lSign: Value[Int],
    rStruct: SBaseStructValue, rLength: Value[Int], rSign: Value[Int]
  ): Value[Int] = {
    val structType = lStruct.st
    assert(rStruct.st.virtualType.isIsomorphicTo(structType.virtualType))
    val prefixLength = cb.memoize(lLength.min(rLength))

    val result = cb.newLocal[Int]("partitionIntervalEndpointCompare")
    val Lafter = CodeLabel()
    val Leq = CodeLabel()
    cb.if_(prefixLength.ceq(0), cb.goto(Leq))
    (0 until (lStruct.st.size min rStruct.st.size)).foreach { idx =>
      val lField = cb.memoize(lStruct.loadField(cb, idx))
      val rField = cb.memoize(rStruct.loadField(cb, idx))
      cb.assign(result,
        cb.emb.ecb.getOrderingFunction(lField.st, rField.st, CodeOrdering.Compare())
          .apply(cb, lField, rField))
      cb.if_(result.cne(0), cb.goto(Lafter))
      if (idx < (lStruct.st.size min rStruct.st.size)) {
        cb.if_(prefixLength.ceq(idx + 1), cb.goto(Leq))
      }
    }

    cb.define(Leq)
    val c = cb.memoize(lLength - rLength)
    val ls = (c <= 0).mux(lSign, 0)
    val rs = (c >= 0).mux(rSign, 0)
    cb.assign(result, ls - rs)

    cb.define(Lafter)
    result
  }

  def partitionIntervalEndpointCompare(cb: EmitCodeBuilder,
    lhs: SBaseStructValue, lSign: Value[Int],
    rhs: SBaseStructValue, rSign: Value[Int]
  ): Value[Int] = {
    val lStruct = lhs.loadField(cb, 0).get(cb).asBaseStruct
    val lLength = lhs.loadField(cb, 1).get(cb).asInt.value
    val rStruct = rhs.loadField(cb, 0).get(cb).asBaseStruct
    val rLength = rhs.loadField(cb, 1).get(cb).asInt.value
    _partitionIntervalEndpointCompare(cb, lStruct, lLength, lSign, rStruct, rLength, rSign)
  }

  def compareStructWithPartitionIntervalEndpoint(cb: EmitCodeBuilder,
    point: SBaseStructValue,
    intervalEndpoint: SBaseStructValue,
    leansRight: Code[Boolean]
  ): Value[Int] = {
    val endpoint = intervalEndpoint.loadField(cb, 0).get(cb).asBaseStruct
    val endpointLength = intervalEndpoint.loadField(cb, 1).get(cb).asInt.value
    val sign = cb.memoize((leansRight.toI << 1) - 1)
    _partitionIntervalEndpointCompare(cb, point, point.st.size, 0, endpoint, endpointLength, sign)
  }

  def compareStructWithPartitionInterval(cb: EmitCodeBuilder,
    point: SBaseStructValue,
    interval: SIntervalValue
  ): Value[Int] = {
    val start = interval.loadStart(cb)
      .get(cb, "partition intervals cannot have missing endpoints")
      .asBaseStruct
    cb.if_(compareStructWithPartitionIntervalEndpoint(cb, point, start, !interval.includesStart) < 0, {
      primitive(const(-1))
    }, {
      val end = interval.loadEnd(cb)
        .get(cb, "partition intervals cannot have missing endpoints")
        .asBaseStruct
      cb.if_(compareStructWithPartitionIntervalEndpoint(cb, point, end, interval.includesEnd) < 0, {
        primitive(const(0))
      }, {
        primitive(const(1))
      })
    }).asInt.value
  }

  def partitionerFindIntervalRange(cb: EmitCodeBuilder, intervals: SIndexableValue, query: SIntervalValue, errorID: Value[Int]): (Value[Int], Value[Int]) = {
    val needleStart = query.loadStart(cb)
      .get(cb, "partitionerFindIntervalRange assumes non-missing interval endpoints", errorID)
      .asBaseStruct
    val needleEnd = query.loadEnd(cb)
      .get(cb, "partitionerFindIntervalRange assumes non-missing interval endpoints", errorID)
      .asBaseStruct

    def ltNeedle(interval: IEmitCode): Code[Boolean] = {
      val intervalVal = interval
        .get(cb, "partitionerFindIntervalRange: partition intervals cannot be missing", errorID)
        .asInterval
      val intervalEnd = intervalVal.loadEnd(cb)
        .get(cb, "partitionerFindIntervalRange assumes non-missing interval endpoints", errorID)
        .asBaseStruct
      val c = partitionIntervalEndpointCompare(cb,
        intervalEnd, cb.memoize((intervalVal.includesEnd.toI << 1) - 1),
        needleStart, cb.memoize(const(1) - (query.includesStart.toI << 1)))
      c <= 0
    }

    def gtNeedle(interval: IEmitCode): Code[Boolean] = {
      val intervalVal = interval
        .get(cb, "partitionerFindIntervalRange: partition intervals cannot be missing", errorID)
        .asInterval
      val intervalStart = intervalVal.loadStart(cb)
        .get(cb, "partitionerFindIntervalRange assumes non-missing interval endpoints", errorID)
        .asBaseStruct
      val c = partitionIntervalEndpointCompare(cb,
        intervalStart, cb.memoize(const(1) - (intervalVal.includesStart.toI << 1)),
        needleEnd, cb.memoize((query.includesEnd.toI << 1) - 1))
      c >= 0
    }

    val compare = BinarySearch.Comparator.fromLtGt(ltNeedle, gtNeedle)

    BinarySearch.equalRange(cb, intervals, compare, ltNeedle, gtNeedle, 0, intervals.loadLength())
  }

  def arrayOfStructFindIntervalRange(cb: EmitCodeBuilder,
    array: SIndexableValue,
    startKey: SBaseStructValue, startLeansRight: Value[Boolean],
    endKey: SBaseStructValue, endLeansRight: Value[Boolean],
    key: IEmitCode => IEmitCode
  ): (Value[Int], Value[Int]) = {
    def ltNeedle(elt: IEmitCode): Code[Boolean] = {
      val eltKey = cb.memoize(key(elt)).get(cb).asBaseStruct
      val c = compareStructWithPartitionIntervalEndpoint(cb, eltKey, startKey, startLeansRight)
      c <= 0
    }
    def gtNeedle(elt: IEmitCode): Code[Boolean] = {
      val eltKey = cb.memoize(key(elt)).get(cb).asBaseStruct
      val c = compareStructWithPartitionIntervalEndpoint(cb, eltKey, endKey, endLeansRight)
      c >= 0
    }
    val compare = BinarySearch.Comparator.fromLtGt(ltNeedle, gtNeedle)

    BinarySearch.equalRange(cb, array, compare, ltNeedle, gtNeedle, 0, array.loadLength())
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
      case (r, cb, rt, interval: SIntervalValue, _) => primitive(interval.includesStart)
    }

    registerSCode1("includesEnd", TInterval(tv("T")), TBoolean, (_: Type, x: SType) =>
      SBoolean
    ) {
      case (r, cb, rt, interval: SIntervalValue, _) => primitive(interval.includesEnd)
    }

    registerIEmitCode2("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(_: Type, intervalT: EmitType, pointT: EmitType) =>
        val intervalST = intervalT.st.asInstanceOf[SInterval]
        val required = intervalT.required && intervalST.pointEmitType.required && pointT.required
        EmitType(SBoolean, required)
    }) { case (cb, r, rt, _, int, point) =>
      IEmitCode.multiFlatMap(cb,
        FastSeq(int.toI, point.toI)
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
      IEmitCode.multiFlatMap(cb, FastSeq(interval1.toI, interval2.toI)) {
        case Seq(interval1: SIntervalValue, interval2: SIntervalValue) =>
        intervalsOverlap(cb, interval1, interval2)
      }
    }

    registerSCode2("sortedNonOverlappingIntervalsContain",
      TArray(TInterval(tv("T"))), tv("T"), TBoolean, (_, _, _) => SBoolean
    ) { case (_, cb, rt, intervals, point, errorID) =>
      val compare = BinarySearch.Comparator.fromCompare { intervalEC =>
        val interval = intervalEC
          .get(cb, "sortedNonOverlappingIntervalsContain assumes non-missing intervals", errorID)
          .asInterval
        intervalPointCompare(cb, interval, point)
          .get(cb, "sortedNonOverlappingIntervalsContain assumes non-missing interval endpoints", errorID)
          .asInt.value
      }

      primitive(BinarySearch.containsOrdered(cb, intervals.asIndexable, compare))
    }

    val partitionEndpointType = TTuple(tv("T"), TInt32)
    val partitionIntervalType = TInterval(partitionEndpointType)
    registerSCode2("partitionerContains",
      TArray(partitionIntervalType), tv("T"), TBoolean,
      (_, _, _) => SBoolean
    ) { case (_, cb, rt, intervals: SIndexableValue, point: SBaseStructValue, errorID) =>
      def ltNeedle(interval: IEmitCode): Code[Boolean] = {
        val intervalVal = interval
          .get(cb, "partitionerFindIntervalRange: partition intervals cannot be missing", errorID)
          .asInterval
        val intervalEnd = intervalVal.loadEnd(cb)
          .get(cb, "partitionerFindIntervalRange assumes non-missing interval endpoints", errorID)
          .asBaseStruct
        val c = compareStructWithPartitionIntervalEndpoint(cb,
          point,
          intervalEnd, intervalVal.includesEnd)
        c > 0
      }
      def gtNeedle(interval: IEmitCode): Code[Boolean] = {
        val intervalVal = interval
          .get(cb, "partitionerFindIntervalRange: partition intervals cannot be missing", errorID)
          .asInterval
        val intervalStart = intervalVal.loadStart(cb)
          .get(cb, "partitionerFindIntervalRange assumes non-missing interval endpoints", errorID)
          .asBaseStruct
        val c = compareStructWithPartitionIntervalEndpoint(cb,
          point,
          intervalStart, !intervalVal.includesStart)
        c < 0
      }
      primitive(BinarySearch.containsOrdered(cb, intervals, ltNeedle, gtNeedle))
    }

    val requiredInt = EmitType(SInt32, true)
    val equalRangeResultType = TTuple(TInt32, TInt32)
    val equalRangeResultSType = SStackStruct(equalRangeResultType, FastSeq(requiredInt, requiredInt))

    registerSCode2("partitionerFindIntervalRange",
      TArray(partitionIntervalType), partitionIntervalType, equalRangeResultType,
      (_, _, _) => equalRangeResultSType
    ) { case (_, cb, rt, intervals: SIndexableValue, query: SIntervalValue, errorID) =>
      val (start, end) = partitionerFindIntervalRange(cb, intervals, query, errorID)
      new SStackStructValue(equalRangeResultSType,
        FastSeq(
          EmitValue.present(primitive(start)),
          EmitValue.present(primitive(end))))
    }

    val endpointT = TTuple(tv("T"), TInt32)
    registerSCode3("pointLessThanPartitionIntervalLeftEndpoint", tv("T"), endpointT, TBoolean, TBoolean, (_, _, _, _) => SBoolean) {
      case (_, cb, _, point: SBaseStructValue, leftPartitionEndpoint: SBaseStructValue, containsStart: SBooleanValue, _) =>
        primitive(cb.memoize(
          compareStructWithPartitionIntervalEndpoint(cb, point, leftPartitionEndpoint, !containsStart.value) < 0))
    }

    registerSCode3("pointLessThanPartitionIntervalRightEndpoint", tv("T"), endpointT, TBoolean, TBoolean, (_, _, _, _) => SBoolean) {
      case (_, cb, _, point: SBaseStructValue, rightPartitionEndpoint: SBaseStructValue, containsEnd: SBooleanValue, _) =>
        primitive(cb.memoize(
          compareStructWithPartitionIntervalEndpoint(cb, point, rightPartitionEndpoint, containsEnd.value) < 0))
    }

    registerSCode2("partitionIntervalContains",
      partitionIntervalType,
      tv("T"), TBoolean, (_, _, _) => SBoolean) {
      case (_, cb, _, interval: SIntervalValue, point: SBaseStructValue, _) =>
        val leftTuple = interval.loadStart(cb).get(cb).asBaseStruct

        val includesLeft = interval.includesStart
        val pointGTLeft = compareStructWithPartitionIntervalEndpoint(cb, point, leftTuple, !includesLeft) > 0

        val isContained = cb.newLocal[Boolean]("partitionInterval_b", pointGTLeft)

        cb.if_(isContained, {
          // check right endpoint
          val rightTuple = interval.loadEnd(cb).get(cb).asBaseStruct

          val includesRight = interval.includesEnd
          val pointLTRight = compareStructWithPartitionIntervalEndpoint(cb, point, rightTuple, includesRight) < 0
          cb.assign(isContained, pointLTRight)
        })

        primitive(isContained)
    }
  }
}

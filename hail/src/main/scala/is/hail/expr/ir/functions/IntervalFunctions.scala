package is.hail.expr.ir.functions

import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SIntervalPointer
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SBooleanValue}
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.virtual._

object IntervalFunctions extends RegistryFunctions {

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
              cb.memoize(includesStart.asBoolean.boolCode(cb)),
              cb.memoize(includesEnd.asBoolean.boolCode(cb))
            )
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
          val pointv = cb.memoize(point.toI(cb), "point")
          val compare = cb.emb.ecb.getOrderingFunction(pointv.st, interval.st.pointType, CodeOrdering.Compare())

          val start = EmitCode.fromI(cb.emb)(cb => interval.loadStart(cb))
          val cmp = cb.newLocal("cmp", compare(cb, pointv, start))
          val contains = cb.newLocal[Boolean]("contains", false)
          cb.ifx(cmp > 0 || (cmp.ceq(0) && interval.includesStart()), {
            val end = EmitCode.fromI(cb.emb)(cb => interval.loadEnd(cb))
            cb.assign(cmp, compare(cb, pointv, end))
            cb.assign(contains, cmp < 0 || (cmp.ceq(0) && interval.includesEnd()))
          })

          primitive(contains)
        }
    }

    registerSCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: SType) => SBoolean) {
      case (r, cb, rt, interval: SIntervalValue, _) =>
        primitive(interval.isEmpty(cb))
    }

    registerSCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, (_: Type, i1t: SType, i2t: SType) => SBoolean) {
      case (r, cb, rt, interval1: SIntervalValue, interval2: SIntervalValue, _) =>
        val compare = cb.emb.ecb.getOrderingFunction(interval1.st.pointType, interval2.st.pointType, CodeOrdering.Compare())

        def isAboveOnNonempty(cb: EmitCodeBuilder, lhs: SIntervalValue, rhs: SIntervalValue): Code[Boolean] = {
          val start = EmitCode.fromI(cb.emb)(cb => lhs.loadStart(cb))
          val end = EmitCode.fromI(cb.emb)(cb => rhs.loadEnd(cb))
          val cmp = cb.newLocal("cmp", compare(cb, start, end))
          cmp > 0 || (cmp.ceq(0) && (!lhs.includesStart() || !rhs.includesEnd()))
        }

        def isBelowOnNonempty(cb: EmitCodeBuilder, lhs: SIntervalValue, rhs: SIntervalValue): Code[Boolean] = {
          val end = EmitCode.fromI(cb.emb)(cb => lhs.loadEnd(cb))
          val start = EmitCode.fromI(cb.emb)(cb => rhs.loadStart(cb))
          val cmp = cb.newLocal("cmp", compare(cb, end, start))
          cmp < 0 || (cmp.ceq(0) && (!lhs.includesEnd() || !rhs.includesStart()))
        }

        primitive(cb.memoize(
          !(interval1.isEmpty(cb) || interval2.isEmpty(cb) ||
            isBelowOnNonempty(cb, interval1, interval2) ||
            isAboveOnNonempty(cb, interval1, interval2))))
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
    registerSCode3("partitionIntervalEndpointGreaterThan", endpointT, tv("T"), TBoolean, TBoolean, (_, _, _, _) => SBoolean) {
      case (_, cb, _, leftPartitionEndpoint: SBaseStructValue, point: SBaseStructValue, containsStart: SBooleanValue, _) =>
        val left = leftPartitionEndpoint.loadField(cb, 0).get(cb).asBaseStruct
        val leftLen = cb.newLocal[Int]("partitionInterval_leftlen", leftPartitionEndpoint.loadField(cb, 1).get(cb).asInt.intCode(cb))

        val c = cb.newLocal[Int]("partitionInterval_c", 0)
        (0 until left.st.size).foreach { idx =>
          cb.ifx(c.ceq(0) && const(idx) < leftLen, {
            val leftField = EmitCode.fromI(cb.emb)(cb => left.loadField(cb, idx))
            val pointField = EmitCode.fromI(cb.emb)(cb => point.loadField(cb, idx))
            cb.assign(c, cb.emb.ecb.getOrderingFunction(leftField.st, pointField.st, CodeOrdering.Compare())
              .apply(cb, leftField, pointField))
          })
        }

        val isContained = cb.newLocal[Boolean]("partitionInterval_b")
        cb.ifx(c.ceq(0),
          cb.assign(isContained, containsStart.boolCode(cb)),
          cb.assign(isContained, c < 0))

        primitive(isContained)
    }

    registerSCode3("partitionIntervalEndpointLessThan", endpointT, tv("T"), TBoolean, TBoolean, (_, _, _, _) => SBoolean) {
      case (_, cb, _, rightPartitionEndpoint: SBaseStructValue, point: SBaseStructValue, containsEnd: SBooleanValue, _) =>
        val right = rightPartitionEndpoint.loadField(cb, 0).get(cb).asBaseStruct
        val rightLen = cb.newLocal[Int]("partitionInterval_rightlen", rightPartitionEndpoint.loadField(cb, 1).get(cb).asInt.intCode(cb))

        val c = cb.newLocal[Int]("partitionInterval_c", 0)
        (0 until right.st.size).foreach { idx =>
          cb.ifx(c.ceq(0) && const(idx) < rightLen, {
            val rightField = EmitCode.fromI(cb.emb)(cb => right.loadField(cb, idx))
            val pointField = EmitCode.fromI(cb.emb)(cb => point.loadField(cb, idx))
            cb.assign(c, cb.emb.ecb.getOrderingFunction(rightField.st, pointField.st, CodeOrdering.Compare())
              .apply(cb, rightField, pointField))
          })
        }

        val isContained = cb.newLocal[Boolean]("partitionInterval_b")
        cb.ifx(c.ceq(0),
          cb.assign(isContained, containsEnd.boolCode(cb)),
          cb.assign(isContained, c > 0))

        primitive(isContained)
    }


    registerSCode2("partitionIntervalContains",
      TStruct("left" -> endpointT, "right" -> endpointT, "includesLeft" -> TBoolean, "includesRight" -> TBoolean),
      tv("T"), TBoolean, (_, _, _) => SBoolean) {
      case (er, cb, _, interval: SBaseStructValue, point: SBaseStructValue, _) =>
        val c = cb.newLocal[Int]("partitionInterval_c", 0)

        val leftTuple = interval.loadField(cb, "left").get(cb).asBaseStruct

        val left = leftTuple.loadField(cb, 0).get(cb).asBaseStruct
        val leftLen = cb.newLocal[Int]("partitionInterval_leftlen", leftTuple.loadField(cb, 1).get(cb).asInt.intCode(cb))

        (0 until left.st.size).foreach { idx =>
          cb.ifx(c.ceq(0) && const(idx) < leftLen, {
            val leftField = EmitCode.fromI(cb.emb)(cb => left.loadField(cb, idx))
            val pointField = EmitCode.fromI(cb.emb)(cb => point.loadField(cb, idx))
            cb.assign(c, cb.emb.ecb.getOrderingFunction(leftField.st, pointField.st, CodeOrdering.Compare())
              .apply(cb, leftField, pointField))
          })
        }

        val isContained = cb.newLocal[Boolean]("partitionInterval_b")
        cb.ifx(c.ceq(0),
          cb.assign(isContained, interval.loadField(cb, "includesLeft").get(cb).asBoolean.boolCode(cb)),
          cb.assign(isContained, c < 0))

        cb.ifx(isContained, {
          // check right endpoint
          val rightTuple = interval.loadField(cb, "right").get(cb).asBaseStruct

          val right = rightTuple.loadField(cb, 0).get(cb).asBaseStruct
          val rightLen = cb.newLocal[Int]("partitionInterval_leftlen", rightTuple.loadField(cb, 1).get(cb).asInt.intCode(cb))

          cb.assign(c, 0)
          (0 until right.st.size).foreach { idx =>
            cb.ifx(c.ceq(0) && const(idx) < rightLen, {
              val rightField = EmitCode.fromI(cb.emb)(cb => right.loadField(cb, idx))
              val pointField = EmitCode.fromI(cb.emb)(cb => point.loadField(cb, idx))
              cb.assign(c, cb.emb.ecb.getOrderingFunction(rightField.st, pointField.st, CodeOrdering.Compare())
                .apply(cb, rightField, pointField))
            })
          }

          cb.ifx(c.ceq(0),
            cb.assign(isContained, interval.loadField(cb, "includesRight").get(cb).asBoolean.boolCode(cb)),
            cb.assign(isContained, c > 0))
        })

        primitive(isContained)
    }
  }
}

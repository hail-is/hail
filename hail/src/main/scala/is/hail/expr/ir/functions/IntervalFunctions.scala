package is.hail.expr.ir.functions

import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.physical.stypes.concrete.SIntervalPointer
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.SBoolean
import is.hail.types.virtual._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerIEmitCode4("Interval", tv("T"), tv("T"), TBoolean, TBoolean, TInterval(tv("T")),
      { case (_: Type, startpt, endpt, includesStartET, includesEndET) =>
        EmitType(PCanonicalInterval(
          InferPType.getCompatiblePType(Seq(startpt.canonicalPType, endpt.canonicalPType)),
          required = includesStartET.required && includesEndET.required
        ).sType, includesStartET.required && includesEndET.required)
      }) {
      case (cb, r, SIntervalPointer(pt: PCanonicalInterval), start, end, includesStart, includesEnd) =>

        includesStart.toI(cb).flatMap(cb) { includesStart =>
          includesEnd.toI(cb).map(cb) { includesEnd =>

            pt.constructFromCodes(cb, r,
              start,
              end,
              EmitCode.present(cb.emb, includesStart),
              EmitCode.present(cb.emb, includesEnd))
          }
        }
    }

    registerIEmitCode1("start", TInterval(tv("T")), tv("T"),
      (_: Type, x: EmitType) => EmitType(x.st.asInstanceOf[SInterval].pointType, x.required && x.st.asInstanceOf[SInterval].pointEmitType.required)) {
      case (cb, r, rt, interval) =>
        interval.toI(cb).flatMap(cb) { case pi: SIntervalCode =>
          val pv = pi.memoize(cb, "interval")
          pv.loadStart(cb)
        }
    }

    registerIEmitCode1("end", TInterval(tv("T")), tv("T"),
      (_: Type, x: EmitType) => EmitType(x.st.asInstanceOf[SInterval].pointType, x.required && x.st.asInstanceOf[SInterval].pointEmitType.required)) {
      case (cb, r, rt, interval) =>
        interval.toI(cb).flatMap(cb) { case pi: SIntervalCode =>
          val pv = pi.memoize(cb, "interval")
          pv.loadEnd(cb)
        }
    }

    registerPCode1("includesStart", TInterval(tv("T")), TBoolean, (_: Type, x: SType) =>
      SBoolean
    ) {
      case (r, cb, rt, interval: SIntervalCode) => primitive(interval.includesStart())
    }

    registerPCode1("includesEnd", TInterval(tv("T")), TBoolean, (_: Type, x: SType) =>
      SBoolean
    ) {
      case (r, cb, rt, interval: SIntervalCode) => primitive(interval.includesEnd())
    }

    registerIEmitCode2("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(_: Type, intervalT: EmitType, _: EmitType) => EmitType(SBoolean, intervalT.required)
    }) {
      case (cb, r, rt, int, point) =>
        int.toI(cb).map(cb) { case (intc: SIntervalCode) =>
          val interval: SIntervalValue = intc.memoize(cb, "interval")
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

    registerPCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: SType) => SBoolean) {
      case (r, cb, rt, interval: SIntervalCode) =>
        val empty = EmitCodeBuilder.scopedCode(r.mb) { cb =>
          val intv = interval.memoize(cb, "interval")
          intv.isEmpty(cb)
        }
        primitive(empty)
    }

    registerPCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, (_: Type, i1t: SType, i2t: SType) => SBoolean) {
      case (r, cb, rt, int1: SIntervalCode, int2: SIntervalCode) =>
        val overlap = EmitCodeBuilder.scopedCode(r.mb) { cb =>
          val interval1 = int1.memoize(cb, "interval1")
          val interval2 = int2.memoize(cb, "interval2")
          val compare = cb.emb.ecb.getOrderingFunction(int1.st.pointType, int2.st.pointType, CodeOrdering.Compare())

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

          !(interval1.isEmpty(cb) || interval2.isEmpty(cb) ||
            isBelowOnNonempty(cb, interval1, interval2) ||
            isAboveOnNonempty(cb, interval1, interval2))
        }
        primitive(overlap)
    }

    registerIR2("sortedNonOverlappingIntervalsContain",
      TArray(TInterval(tv("T"))), tv("T"), TBoolean) { case (_, intervals, value) =>
      val uid = genUID()
      val uid2 = genUID()
      Let(uid, LowerBoundOnOrderedCollection(intervals, value, onKey = true),
        (Let(uid2, Ref(uid, TInt32) - I32(1), (Ref(uid2, TInt32) >= 0)
          && invoke("contains", TBoolean, ArrayRef(intervals, Ref(uid2, TInt32)), value)))
          || ((Ref(uid, TInt32) < ArrayLen(intervals))
          && invoke("contains", TBoolean, ArrayRef(intervals, Ref(uid, TInt32)), value)))
    }


    val endpointT = TTuple(tv("T"), TInt32)
    registerIR2("partitionIntervalContains",
      TStruct("left" -> endpointT, "right" -> endpointT, "includesLeft" -> TBoolean, "includesRight" -> TBoolean),
      tv("T"), TBoolean) {
      case (_, interval, point) =>

        def compareStructs(left: IR, right: IR): IR = {
          bindIRs(left, right) { case Seq(lTuple, r) =>
            bindIRs(GetTupleElement(lTuple, 0), GetTupleElement(lTuple, 1)) {
              case Seq(lValue, lLen) =>
                val ts = lValue.typ.asInstanceOf[TStruct]
                assert(r.typ == ts)
                ts.fields.foldRight[IR](I32(0)) { case (f, acc) =>
                  If(
                    lLen ceq f.index,
                    0,
                    bindIR(ApplyComparisonOp(Compare(f.typ), GetField(lValue, f.name), GetField(r, f.name))) { c =>
                      If(c.cne(0), c, acc)
                    })
                }
            }
          }
        }

        bindIRs(point, GetField(interval, "left"), GetField(interval, "right")) { case Seq(point, l, r) =>


          val gtEqLeft = bindIR(compareStructs(l, point)) { lc =>
            (lc <= 0) && ((lc < 0) || GetField(interval, "includesLeft"))
          }

          val ltEqRight = bindIR(compareStructs(r, point)) { rc =>
            (rc >= 0) && ((rc > 0) || GetField(interval, "includesRight"))
          }
          gtEqLeft && ltEqRight
        }
    }
  }
}

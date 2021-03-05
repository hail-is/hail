package is.hail.expr.ir.functions

import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.virtual._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerIEmitCode4("Interval", tv("T"), tv("T"), TBoolean, TBoolean, TInterval(tv("T")),
      { case (_: Type, startpt, endpt, includesStartPT, includesEndPT) =>
        PCanonicalInterval(
          InferPType.getCompatiblePType(Seq(startpt, endpt)),
          required = includesStartPT.required && includesEndPT.required
        )
      }) {
      case (cb, r, rt: PCanonicalInterval, start, end, includesStart, includesEnd) =>

        includesStart.toI(cb).flatMap(cb) { includesStart =>
          includesEnd.toI(cb).map(cb) { includesEnd =>

            rt.constructFromCodes(cb, r,
              start,
              end,
              EmitCode.present(cb.emb, includesStart),
              EmitCode.present(cb.emb, includesEnd))
          }
        }
    }

    registerIEmitCode1("start", TInterval(tv("T")), tv("T"),
      (_: Type, x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (cb, r, rt, interval) =>
        interval.toI(cb).flatMap(cb) { case pi: PIntervalCode =>
          val pv = pi.memoize(cb, "interval")
          pv.loadStart(cb).typecast[PCode]
        }
    }

    registerIEmitCode1("end", TInterval(tv("T")), tv("T"),
      (_: Type, x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (cb, r, rt, interval) =>
        interval.toI(cb).flatMap(cb) { case pi: PIntervalCode =>
          val pv = pi.memoize(cb, "interval")
          pv.loadEnd(cb).typecast[PCode]
        }
    }

    registerPCode1("includesStart", TInterval(tv("T")), TBoolean, (_: Type, x: PType) =>
      PBoolean(x.required)
    ) {
      case (r, cb, rt, interval: PIntervalCode) => PCode(rt, interval.includesStart())
    }

    registerPCode1("includesEnd", TInterval(tv("T")), TBoolean, (_: Type, x: PType) =>
      PBoolean(x.required)
    ) {
      case (r, cb, rt, interval: PIntervalCode) => PCode(rt, interval.includesEnd())
    }

    registerIEmitCode2("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(_: Type, intervalT: PInterval, _: PType) => PBoolean(intervalT.required)
    }) {
      case (cb, r, rt, int, point) =>
        int.toI(cb).map(cb) { case (intc: PIntervalCode) =>
          val interval: PIntervalValue = intc.memoize(cb, "interval")
          val pointv = cb.memoize(point.toI(cb), "point")
          val compare = cb.emb.ecb.getOrderingFunction(pointv.st, interval.st.pointType, CodeOrdering.Compare())

          val start = EmitCode.fromI(cb.emb)(cb => interval.loadStart(cb).typecast[PCode])
          val cmp = cb.newLocal("cmp", compare(cb, pointv, start))
          val contains = cb.newLocal[Boolean]("contains", false)
          cb.ifx(cmp > 0 || (cmp.ceq(0) && interval.includesStart()), {
            val end = EmitCode.fromI(cb.emb)(cb => interval.loadEnd(cb).typecast[PCode])
            cb.assign(cmp, compare(cb, pointv, end))
            cb.assign(contains, cmp < 0 || (cmp.ceq(0) && interval.includesEnd()))
          })

          PCode(rt, contains)
        }
    }

    registerPCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: PType) => PBoolean(pt.required)) {
      case (r, cb, rt, interval: PIntervalCode) =>
        val empty = EmitCodeBuilder.scopedCode(r.mb) { cb =>
          val intv = interval.memoize(cb, "interval")
          intv.isEmpty(cb)
        }
        PCode(rt, empty)
    }

    registerPCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, (_: Type, i1t: PType, i2t: PType) => PBoolean(i1t.required && i2t.required)) {
      case (r, cb, rt, int1: PIntervalCode, int2: PIntervalCode) =>
        val overlap = EmitCodeBuilder.scopedCode(r.mb) { cb =>
          val interval1 = int1.memoize(cb, "interval1")
          val interval2 = int2.memoize(cb, "interval2")
          val compare = cb.emb.ecb.getOrderingFunction(int1.st.pointType, int2.st.pointType, CodeOrdering.Compare())

          def isAboveOnNonempty(cb: EmitCodeBuilder, lhs: PIntervalValue, rhs: PIntervalValue): Code[Boolean] = {
            val start = EmitCode.fromI(cb.emb)(cb => lhs.loadStart(cb).typecast[PCode])
            val end = EmitCode.fromI(cb.emb)(cb => rhs.loadEnd(cb).typecast[PCode])
            val cmp = cb.newLocal("cmp", compare(cb, start, end))
            cmp > 0 || (cmp.ceq(0) && (!lhs.includesStart() || !rhs.includesEnd()))
          }

          def isBelowOnNonempty(cb: EmitCodeBuilder, lhs: PIntervalValue, rhs: PIntervalValue): Code[Boolean] = {
            val end = EmitCode.fromI(cb.emb)(cb => lhs.loadEnd(cb).typecast[PCode])
            val start = EmitCode.fromI(cb.emb)(cb => rhs.loadStart(cb).typecast[PCode])
            val cmp = cb.newLocal("cmp", compare(cb, end, start))
            cmp < 0 || (cmp.ceq(0) && (!lhs.includesEnd() || !rhs.includesStart()))
          }

          !(interval1.isEmpty(cb) || interval2.isEmpty(cb) ||
            isBelowOnNonempty(cb, interval1, interval2) ||
            isAboveOnNonempty(cb, interval1, interval2))
        }
        PCode(rt, overlap)
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

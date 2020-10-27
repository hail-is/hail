package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.types.virtual.{TArray, TBoolean, TInt32, TInterval, TString, TStruct, TTuple, Type}
import is.hail.utils._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerEmitCode4("Interval", tv("T"), tv("T"), TBoolean, TBoolean, TInterval(tv("T")),
      { case (_: Type, startpt, endpt, includesStartPT, includesEndPT) =>
        PCanonicalInterval(
          InferPType.getCompatiblePType(Seq(startpt, endpt)),
          required = includesStartPT.required && includesEndPT.required
        )
      }) {
      case (r, rt, start, end, includesStart, includesEnd) =>
        val srvb = new StagedRegionValueBuilder(r, rt)

        val mv = r.mb.newLocal[Boolean]()
        val vv = r.mb.newLocal[Long]()

        val ctor = Code(
          mv := includesStart.m || includesEnd.m,
          vv := 0L,
          mv.mux(
            Code._empty,
            Code(FastIndexedSeq(
              srvb.start(),
              start.m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(start.pt)(start.v)),
              srvb.advance(),
              end.m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(end.pt)(end.v)),
              srvb.advance(),
              srvb.addBoolean(includesStart.value[Boolean]),
              srvb.advance(),
              srvb.addBoolean(includesEnd.value[Boolean]),
              srvb.advance(),
              vv := srvb.offset))),
          Code._empty)

        EmitCode(
          Code(start.setup, end.setup, includesStart.setup, includesEnd.setup, ctor),
          mv,
          PCode(rt, vv))
    }

    registerIEmitCode1("start", TInterval(tv("T")), tv("T"),
      (_: Type, x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (cb, r, rt, interval) =>
        interval().flatMap(cb) { case pi: PIntervalCode =>
          val pv = pi.memoize(cb, "interval")
          pv.loadStart(cb)
        }
    }

    registerIEmitCode1("end", TInterval(tv("T")), tv("T"),
      (_: Type, x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (cb, r, rt, interval) =>
        interval().flatMap(cb) { case pi: PIntervalCode =>
          val pv = pi.memoize(cb, "interval")
          pv.loadEnd(cb)
        }
    }

    registerPCode1("includesStart", TInterval(tv("T")), TBoolean, (_: Type, x: PType) =>
      PBoolean(x.required)
    ) {
      case (r, rt, interval: PIntervalCode) => PCode(rt, interval.includesStart())
    }

    registerPCode1("includesEnd", TInterval(tv("T")), TBoolean, (_: Type, x: PType) =>
      PBoolean(x.required)
    ) {
      case (r, rt, interval: PIntervalCode) => PCode(rt, interval.includesEnd())
    }

    registerIEmitCode2("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(_: Type, intervalT: PInterval, _: PType) => PBoolean(intervalT.required)
    }) {
      case (cb, r, rt, int, point) =>
        int().map(cb) { case (intc: PIntervalCode) =>
          val interval: PIntervalValue = intc.memoize(cb, "interval")
          val pointv = cb.memoize(point(), "point")
          val compare = cb.emb.getCodeOrdering(pointv.pt, interval.pt.pointType, CodeOrdering.Compare())

          val start = EmitCode.fromI(cb.emb)(interval.loadStart(_))
          cb += start.setup
          val cmp = cb.newLocal("cmp", compare(pointv.m -> pointv.v, start.m -> start.v))
          val contains = cb.newLocal[Boolean]("contains", false)
          cb.ifx(cmp > 0 || (cmp.ceq(0) && interval.includesStart()), {
            val end = EmitCode.fromI(cb.emb)(interval.loadEnd(_))
            cb += end.setup
            cb.assign(cmp, compare(pointv.m -> pointv.v, end.m -> end.v))
            cb.assign(contains, cmp < 0 || (cmp.ceq(0) && interval.includesEnd()))
          })

          PCode(rt, contains)
        }
    }

    registerPCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: PType) => PBoolean(pt.required)) {
      case (r, rt, interval: PIntervalCode) =>
        val empty = EmitCodeBuilder.scopedCode(r.mb) { cb =>
          val intv = interval.memoize(cb, "interval")
          intv.isEmpty(cb)
        }
        PCode(rt, empty)
    }

    registerPCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, (_: Type, i1t: PType, i2t: PType) => PBoolean(i1t.required && i2t.required)) {
      case (r, rt, int1: PIntervalCode, int2: PIntervalCode) =>
        val overlap = EmitCodeBuilder.scopedCode(r.mb) { cb =>
          val interval1 = int1.memoize(cb, "interval1")
          val interval2 = int2.memoize(cb, "interval2")
          val compare = cb.emb.getCodeOrdering(int1.pt.pointType, int2.pt.pointType, CodeOrdering.Compare())

          def isAboveOnNonempty(cb: EmitCodeBuilder, lhs: PIntervalValue, rhs: PIntervalValue): Code[Boolean] = {
            val cmp = r.mb.newLocal[Int]()
            val start = EmitCode.fromI(cb.emb)(lhs.loadStart(_))
            val end = EmitCode.fromI(cb.emb)(rhs.loadEnd(_))
            cb += start.setup
            cb += end.setup
            cb.assign(cmp, compare(start.m -> start.v, end.m -> end.v))
            cmp > 0 || (cmp.ceq(0) && (!lhs.includesStart() || !rhs.includesEnd()))
          }

          def isBelowOnNonempty(cb: EmitCodeBuilder, lhs: PIntervalValue, rhs: PIntervalValue): Code[Boolean] = {
            val start = EmitCode.fromI(cb.emb)(rhs.loadStart(_))
            val end = EmitCode.fromI(cb.emb)(lhs.loadEnd(_))
            cb += start.setup
            cb += end.setup
            val cmp = r.mb.newLocal[Int]()
            cb.newLocal("cmp", compare(start.m -> start.v, end.m -> end.v))
            cmp < 0 || (cmp.ceq(0) && (!lhs.includesEnd() || !rhs.includesStart()))
          }

          interval1.isEmpty(cb) || interval2.isEmpty(cb) ||
            isBelowOnNonempty(cb, interval1, interval2) ||
            isAboveOnNonempty(cb, interval1, interval2)
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

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

    registerEmitCode1("start", TInterval(tv("T")), tv("T"),
      (_: Type, x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (r, rt, interval) =>
        val intervalT = interval.pt.asInstanceOf[PInterval]
        val iv = r.mb.newLocal[Long]()
        EmitCode(
          Code(interval.setup, iv.storeAny(defaultValue(intervalT))),
          interval.m || !Code(iv := interval.value[Long], intervalT.startDefined(iv)),
          PCode(rt, Region.loadIRIntermediate(intervalT.pointType)(intervalT.startOffset(iv)))
        )
    }

    registerEmitCode1("end", TInterval(tv("T")), tv("T"),
      (_: Type, x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (r, rt, interval) =>
        val intervalT = interval.pt.asInstanceOf[PInterval]
        val iv = r.mb.newLocal[Long]()
        EmitCode(
          Code(interval.setup, iv.storeAny(defaultValue(intervalT))),
          interval.m || !Code(iv := interval.value[Long], intervalT.endDefined(iv)),
          PCode(rt, Region.loadIRIntermediate(intervalT.pointType)(intervalT.endOffset(iv)))
        )
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

    registerEmitCode2("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(_: Type, intervalT: PInterval, _: PType) => PBoolean(intervalT.required)
    }) {
      case (r, rt, int, point) =>
        val mPoint = r.mb.newLocal[Boolean]()
        val vPoint = r.mb.newLocal()(typeToTypeInfo(point.pt))

        val cmp = r.mb.newLocal[Int]()
        val interval = new IRInterval(r, int.pt.asInstanceOf[PInterval], int.value[Long])
        val compare = interval.ordering(CodeOrdering.Compare())

        val contains = Code(
          interval.storeToLocal,
          mPoint := point.m,
          vPoint.storeAny(point.v),
          cmp := compare((mPoint, vPoint), interval.start),
          (cmp > 0 || (cmp.ceq(0) && interval.includesStart)) && Code(
            cmp := compare((mPoint, vPoint), interval.end),
            cmp < 0 || (cmp.ceq(0) && interval.includesEnd)))

        EmitCode(
          Code(int.setup, point.setup),
          int.m,
          PCode(rt, contains))
    }

    registerCode1("isEmpty", TInterval(tv("T")), TBoolean, (_: Type, pt: PType) => PBoolean(pt.required)) {
      case (r, rt, (intervalT: PInterval, intOff)) =>
        val interval = new IRInterval(r, intervalT, intOff)

        Code(
          interval.storeToLocal,
          interval.isEmpty
        )
    }

    registerCode2("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, (_: Type, i1t: PType, i2t: PType) => PBoolean(i1t.required && i2t.required)) {
      case (r, rt, (i1t: PInterval, iOff1), (i2t: PInterval, iOff2)) =>
        val interval1 = new IRInterval(r, i1t, iOff1)
        val interval2 = new IRInterval(r, i2t, iOff2)

        Code(
          interval1.storeToLocal,
          interval2.storeToLocal,
          !(interval1.isEmpty || interval2.isEmpty ||
            interval1.isBelowOnNonempty(interval2) ||
            interval1.isAboveOnNonempty(interval2))
        )
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

class IRInterval(r: EmitRegion, typ: PInterval, value: Code[Long]) {
  val ref: LocalRef[Long] = r.mb.newLocal[Long]()
  val region: Code[Region] = r.region

  def ordering(op: CodeOrdering.Op): ((Code[Boolean], Code[_]), (Code[Boolean], Code[_])) => Code[op.ReturnType] =
    r.mb.getCodeOrdering(typ.pointType, op)(_, _)

  def storeToLocal: Code[Unit] = ref := value

  def start: (Code[Boolean], Code[_]) =
    (!typ.startDefined(ref), Region.getIRIntermediate(typ.pointType)(typ.startOffset(ref)))
  def end: (Code[Boolean], Code[_]) =
    (!typ.endDefined(ref), Region.getIRIntermediate(typ.pointType)(typ.endOffset(ref)))
  def includesStart: Code[Boolean] = typ.includesStart(ref)
  def includesEnd: Code[Boolean] = typ.includesEnd(ref)

  def isEmpty: Code[Boolean] = {
    val gt = ordering(CodeOrdering.Gt())
    val gteq = ordering(CodeOrdering.Gteq())

    (includesStart && includesEnd).mux(
      gt(start, end),
      gteq(start, end))
  }

  def isAboveOnNonempty(other: IRInterval): Code[Boolean] = {
    val cmp = r.mb.newLocal[Int]()
    val compare = ordering(CodeOrdering.Compare())
    Code(
      cmp := compare(start, other.end),
      cmp > 0 || (cmp.ceq(0) && (!includesStart || !other.includesEnd)))
  }

  def isBelowOnNonempty(other: IRInterval): Code[Boolean] = {
    val cmp = r.mb.newLocal[Int]()
    val compare = ordering(CodeOrdering.Compare())
    Code(
      cmp := compare(end, other.start),
      cmp < 0 || (cmp.ceq(0) && (!includesEnd || !other.includesStart)))
  }
}

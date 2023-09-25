package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitSettable, EmitValue, IEmitCode}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.{SInterval, SIntervalValue}
import is.hail.types.virtual.{TInterval, Type}
import is.hail.utils.FastSeq

object SStackInterval {
  def construct(start: EmitValue, end: EmitValue, includesStart: Value[Boolean], includesEnd: Value[Boolean]): SStackIntervalValue = {
    assert(start.emitType == end.emitType)
    new SStackIntervalValue(SStackInterval(start.emitType), start, end, includesStart, includesEnd)
  }
}

final case class SStackInterval(pointEmitType: EmitType) extends SInterval {

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value match {
      case value: SStackIntervalValue => new SStackIntervalValue(this, 
        pointEmitType.coerceOrCopy(cb, region, value.start, deepCopy),
        pointEmitType.coerceOrCopy(cb, region, value.end, deepCopy),
        value.includesStart,
        value.includesEnd
      )
      case value: SIntervalValue =>
        new SStackIntervalValue(this,
          pointEmitType.coerceOrCopy(cb, region, cb.memoize(value.loadStart(cb)), deepCopy),
          pointEmitType.coerceOrCopy(cb, region, cb.memoize(value.loadEnd(cb)), deepCopy),
          value.includesStart,
          value.includesEnd
        ) 
    }


  override def castRename(t: Type): SType = SStackInterval(pointEmitType.copy(st = pointType.castRename(t.asInstanceOf[TInterval].pointType)))

  override lazy val virtualType: Type = TInterval(pointEmitType.virtualType)

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = {
    val pointTypes = pointEmitType.settableTupleTypes
    pointTypes ++ pointTypes ++ FastSeq(BooleanInfo, BooleanInfo)
  }

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SStackIntervalSettable = {
    val pointNSettables = pointEmitType.nSettables
    assert(settables.length == 2 * pointNSettables + 2)
    new SStackIntervalSettable(this, 
      pointEmitType.fromSettables(settables.slice(0, pointNSettables)),
      pointEmitType.fromSettables(settables.slice(pointNSettables, 2 * pointNSettables)),
      coerce[Boolean](settables(pointNSettables * 2)),
      coerce[Boolean](settables(pointNSettables * 2 + 1)))
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SStackIntervalValue = {
    val pointNValues = pointEmitType.nSettables
    assert(values.length == 2 * pointNValues + 2)
    new SStackIntervalValue(this,
      pointEmitType.fromValues(values.slice(0, pointNValues)),
      pointEmitType.fromValues(values.slice(pointNValues, 2 * pointNValues)),
      coerce[Boolean](values(pointNValues * 2)),
      coerce[Boolean](values(pointNValues * 2 + 1)))
  }

  override def pointType: SType = pointEmitType.st

  override def storageType(): PType = pointEmitType.storageType

  override def copiedType: SType = SStackInterval(pointEmitType.copiedType)

  override def containsPointers: Boolean = pointType.containsPointers
}

class SStackIntervalValue(
  val st: SStackInterval,
  val start: EmitValue,
  val end: EmitValue,
  val includesStart: Value[Boolean],
  val includesEnd: Value[Boolean]
) extends SIntervalValue {
  require(start.emitType == end.emitType && start.emitType == st.pointEmitType)
  override lazy val valueTuple: IndexedSeq[Value[_]] = start.valueTuple() ++ end.valueTuple() ++ FastSeq(includesStart, includesEnd)

  override def loadStart(cb: EmitCodeBuilder): IEmitCode = start.toI(cb)

  override def startDefined(cb: EmitCodeBuilder): Value[Boolean] = start.m

  override def loadEnd(cb: EmitCodeBuilder): IEmitCode = end.toI(cb)

  override def endDefined(cb: EmitCodeBuilder): Value[Boolean] = end.m
}

final class SStackIntervalSettable(
  override val st: SStackInterval,
  override val start: EmitSettable,
  override val end: EmitSettable,
  override val includesStart: Settable[Boolean],
  override val includesEnd: Settable[Boolean]
) extends SStackIntervalValue(st, start, end, includesStart, includesEnd) with SSettable {
  override lazy val settableTuple: IndexedSeq[Settable[_]] = start.settableTuple() ++ end.settableTuple() ++ FastSeq(includesStart, includesEnd)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SStackIntervalValue =>
      cb.assign(start, v.start)
      cb.assign(end, v.end)
      cb.assign(includesStart, v.includesStart)
      cb.assign(includesEnd, v.includesEnd)
  }
}

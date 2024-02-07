package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{BooleanInfo, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PInterval, PType}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.{SInterval, SIntervalValue}
import is.hail.types.virtual.Type
import is.hail.utils.FastSeq

final case class SIntervalPointer(pType: PInterval) extends SInterval {
  require(!pType.required)

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    value match {
      case value: SIntervalValue =>
        new SIntervalPointerValue(
          this,
          pType.store(cb, region, value, deepCopy),
          value.includesStart,
          value.includesEnd,
        )
    }

  override def castRename(t: Type): SType =
    SIntervalPointer(pType.deepRename(t).asInstanceOf[PInterval])

  override lazy val virtualType: Type = pType.virtualType

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] =
    FastSeq(LongInfo, BooleanInfo, BooleanInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SIntervalPointerSettable = {
    val IndexedSeq(
      a: Settable[Long @unchecked],
      includesStart: Settable[Boolean @unchecked],
      includesEnd: Settable[Boolean @unchecked],
    ) = settables
    assert(a.ti == LongInfo)
    assert(includesStart.ti == BooleanInfo)
    assert(includesEnd.ti == BooleanInfo)
    new SIntervalPointerSettable(this, a, includesStart, includesEnd)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SIntervalPointerValue = {
    val IndexedSeq(
      a: Value[Long @unchecked],
      includesStart: Value[Boolean @unchecked],
      includesEnd: Value[Boolean @unchecked],
    ) = values
    assert(a.ti == LongInfo)
    assert(includesStart.ti == BooleanInfo)
    assert(includesEnd.ti == BooleanInfo)
    new SIntervalPointerValue(this, a, includesStart, includesEnd)
  }

  override def pointType: SType = pType.pointType.sType
  override def pointEmitType: EmitType = EmitType(pType.pointType.sType, pType.pointType.required)

  override def storageType(): PType = pType

  override def copiedType: SType = SIntervalPointer(pType.copiedType.asInstanceOf[PInterval])

  override def containsPointers: Boolean = pType.containsPointers

  override def isIsomorphicTo(st: SType): Boolean =
    st match {
      case p: SIntervalPointer =>
        pointType isIsomorphicTo p.pointType

      case _ =>
        false
    }
}

class SIntervalPointerValue(
  val st: SIntervalPointer,
  val a: Value[Long],
  val includesStart: Value[Boolean],
  val includesEnd: Value[Boolean],
) extends SIntervalValue {
  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a, includesStart, includesEnd)

  val pt: PInterval = st.pType

  override def loadStart(cb: EmitCodeBuilder): IEmitCode =
    IEmitCode(cb, !pt.startDefined(cb, a), pt.pointType.loadCheapSCode(cb, pt.loadStart(a)))

  override def startDefined(cb: EmitCodeBuilder): Value[Boolean] =
    pt.startDefined(cb, a)

  override def loadEnd(cb: EmitCodeBuilder): IEmitCode =
    IEmitCode(cb, !pt.endDefined(cb, a), pt.pointType.loadCheapSCode(cb, pt.loadEnd(a)))

  override def endDefined(cb: EmitCodeBuilder): Value[Boolean] =
    pt.endDefined(cb, a)
}

object SIntervalPointerSettable {
  def apply(sb: SettableBuilder, st: SIntervalPointer, name: String): SIntervalPointerSettable =
    new SIntervalPointerSettable(
      st,
      sb.newSettable[Long](s"${name}_a"),
      sb.newSettable[Boolean](s"${name}_includes_start"),
      sb.newSettable[Boolean](s"${name}_includes_end"),
    )
}

final class SIntervalPointerSettable(
  st: SIntervalPointer,
  override val a: Settable[Long],
  override val includesStart: Settable[Boolean],
  override val includesEnd: Settable[Boolean],
) extends SIntervalPointerValue(st, a, includesStart, includesEnd) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a, includesStart, includesEnd)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SIntervalPointerValue =>
      cb.assign(a, v.a)
      cb.assign(includesStart, v.includesStart)
      cb.assign(includesEnd, v.includesEnd)
  }
}

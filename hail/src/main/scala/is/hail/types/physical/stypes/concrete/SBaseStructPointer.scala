package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PBaseStruct, PType}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructSettable, SBaseStructValue}
import is.hail.types.virtual.{TBaseStruct, Type}
import is.hail.utils.FastSeq

final case class SBaseStructPointer(pType: PBaseStruct) extends SBaseStruct {
  require(!pType.required)
  override def size: Int = pType.size

  override lazy val virtualType: TBaseStruct = pType.virtualType.asInstanceOf[TBaseStruct]

  override def castRename(t: Type): SType =
    SBaseStructPointer(pType.deepRename(t).asInstanceOf[PBaseStruct])

  override def fieldIdx(fieldName: String): Int = pType.fieldIdx(fieldName)

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    new SBaseStructPointerValue(this, pType.store(cb, region, value, deepCopy))

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SBaseStructPointerSettable = {
    val IndexedSeq(a: Settable[Long @unchecked]) = settables
    assert(a.ti == LongInfo)
    new SBaseStructPointerSettable(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SBaseStructPointerValue = {
    val IndexedSeq(a: Value[Long @unchecked]) = values
    assert(a.ti == LongInfo)
    new SBaseStructPointerValue(this, a)
  }

  def canonicalPType(): PType = pType

  override val fieldTypes: IndexedSeq[SType] = pType.types.map(_.sType)

  override val fieldEmitTypes: IndexedSeq[EmitType] =
    pType.types.map(t => EmitType(t.sType, t.required))

  override def containsPointers: Boolean = pType.containsPointers

  override def storageType(): PType = pType

  override def copiedType: SType = SBaseStructPointer(pType.copiedType.asInstanceOf[PBaseStruct])
}

class SBaseStructPointerValue(
  val st: SBaseStructPointer,
  val a: Value[Long],
) extends SBaseStructValue {
  val pt: PBaseStruct = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a)

  override def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode =
    IEmitCode(
      cb,
      pt.isFieldMissing(cb, a, fieldIdx),
      pt.fields(fieldIdx).typ.loadCheapSCode(cb, pt.loadField(a, fieldIdx)),
    )

  override def isFieldMissing(cb: EmitCodeBuilder, fieldIdx: Int): Value[Boolean] =
    pt.isFieldMissing(cb, a, fieldIdx)
}

object SBaseStructPointerSettable {
  def apply(sb: SettableBuilder, st: SBaseStructPointer, name: String): SBaseStructPointerSettable =
    new SBaseStructPointerSettable(st, sb.newSettable(name))
}

final class SBaseStructPointerSettable(
  st: SBaseStructPointer,
  override val a: Settable[Long],
) extends SBaseStructPointerValue(st, a) with SBaseStructSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(a, v.asInstanceOf[SBaseStructPointerValue].a)
}

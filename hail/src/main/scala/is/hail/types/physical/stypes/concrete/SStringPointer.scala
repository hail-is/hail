package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PString, PType}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SString, SStringValue}
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.virtual.Type
import is.hail.utils.FastSeq

final case class SStringPointer(pType: PString) extends SString {
  require(!pType.required)

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    new SStringPointerValue(this, pType.store(cb, region, value, deepCopy))

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SStringPointerSettable = {
    val IndexedSeq(a: Settable[Long @unchecked]) = settables
    assert(a.ti == LongInfo)
    new SStringPointerSettable(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SStringPointerValue = {
    val IndexedSeq(a: Value[Long @unchecked]) = values
    assert(a.ti == LongInfo)
    new SStringPointerValue(this, a)
  }

  override def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String])
    : SStringPointerValue =
    new SStringPointerValue(this, pType.allocateAndStoreString(cb, r, s))

  override def storageType(): PType = pType

  override def copiedType: SType = SStringPointer(pType.copiedType.asInstanceOf[PString])

  override def containsPointers: Boolean = pType.containsPointers

  override def isIsomorphicTo(st: SType): Boolean =
    st match {
      case p: SStringPointer =>
        pType == p.pType

      case _ =>
        false
    }
}

class SStringPointerValue(val st: SStringPointer, val a: Value[Long]) extends SStringValue {
  val pt: PString = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a)

  def binaryRepr(): SBinaryPointerValue =
    new SBinaryPointerValue(SBinaryPointer(st.pType.binaryRepresentation), a)

  def loadLength(cb: EmitCodeBuilder): Value[Int] =
    cb.memoize(pt.loadLength(a))

  def loadString(cb: EmitCodeBuilder): Value[String] =
    cb.memoize(pt.loadString(a))

  def toBytes(cb: EmitCodeBuilder): SBinaryPointerValue =
    new SBinaryPointerValue(SBinaryPointer(pt.binaryRepresentation), a)

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value =
    this.binaryRepr().sizeToStoreInBytes(cb)
}

object SStringPointerSettable {
  def apply(sb: SettableBuilder, st: SStringPointer, name: String): SStringPointerSettable =
    new SStringPointerSettable(st, sb.newSettable[Long](s"${name}_a"))
}

final class SStringPointerSettable(st: SStringPointer, override val a: Settable[Long])
    extends SStringPointerValue(st, a) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(a, v.asInstanceOf[SStringPointerValue].a)

  override def binaryRepr(): SBinaryPointerSettable =
    new SBinaryPointerSettable(SBinaryPointer(st.pType.binaryRepresentation), a)
}

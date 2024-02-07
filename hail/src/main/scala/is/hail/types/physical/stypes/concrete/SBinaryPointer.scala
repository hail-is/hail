package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PBinary, PType}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue}
import is.hail.types.virtual.Type
import is.hail.utils._

final case class SBinaryPointer(pType: PBinary) extends SBinary {
  require(!pType.required)

  override lazy val virtualType: Type = pType.virtualType

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    new SBinaryPointerValue(this, pType.store(cb, region, value, deepCopy))

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Value[Long]): SValue =
    if (pt == this.pType)
      new SBinaryPointerValue(this, addr)
    else
      coerceOrCopy(cb, region, pt.loadCheapSCode(cb, addr), deepCopy = false)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SBinaryPointerSettable = {
    val IndexedSeq(a: Settable[Long @unchecked]) = settables
    assert(a.ti == LongInfo)
    new SBinaryPointerSettable(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SBinaryPointerValue = {
    val IndexedSeq(a: Value[Long @unchecked]) = values
    assert(a.ti == LongInfo)
    new SBinaryPointerValue(this, a)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SBinaryPointer(pType.copiedType.asInstanceOf[PBinary])

  override def containsPointers: Boolean = pType.containsPointers

  override def castRename(t: Type): SType = this

  override def isIsomorphicTo(st: SType): Boolean =
    st match {
      case p: SBinaryPointer =>
        pType == p.pType

      case _ =>
        false
    }
}

class SBinaryPointerValue(
  val st: SBinaryPointer,
  val a: Value[Long],
) extends SBinaryValue {
  private val pt: PBinary = st.pType

  def bytesAddress(): Code[Long] = st.pType.bytesAddress(a)

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a)

  override def loadLength(cb: EmitCodeBuilder): Value[Int] =
    cb.memoize(pt.loadLength(a))

  override def loadBytes(cb: EmitCodeBuilder): Value[Array[Byte]] =
    cb.memoize(pt.loadBytes(a))

  override def loadByte(cb: EmitCodeBuilder, i: Code[Int]): Value[Byte] =
    cb.memoize(Region.loadByte(pt.bytesAddress(a) + i.toL))
}

object SBinaryPointerSettable {
  def apply(sb: SettableBuilder, st: SBinaryPointer, name: String): SBinaryPointerSettable =
    new SBinaryPointerSettable(st, sb.newSettable[Long](name))
}

final class SBinaryPointerSettable(
  st: SBinaryPointer,
  override val a: Settable[Long],
) extends SBinaryPointerValue(st, a) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(a, v.asInstanceOf[SBinaryPointerValue].a)
}

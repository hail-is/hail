package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryCode, SBinaryValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PBinary, PType}
import is.hail.types.virtual.Type
import is.hail.utils._


final case class SBinaryPointer(pType: PBinary) extends SBinary {
  require(!pType.required)

  override lazy val virtualType: Type = pType.virtualType
  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    new SBinaryPointerValue(this, pType.store(cb, region, value, deepCopy))
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    if (pt == this.pType)
      new SBinaryPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.loadCheapSCode(cb, addr), deepCopy = false).get
  }

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SBinaryPointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    new SBinaryPointerSettable(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SBinaryPointerValue = {
    val IndexedSeq(a: Value[Long@unchecked]) = values
    assert(a.ti == LongInfo)
    new SBinaryPointerValue(this, a)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SBinaryPointer(pType.copiedType.asInstanceOf[PBinary])

  override def containsPointers: Boolean = pType.containsPointers

  override def castRename(t: Type): SType = this
}

class SBinaryPointerValue(
  val st: SBinaryPointer,
  val a: Value[Long]
) extends SBinaryValue {
  private val pt: PBinary = st.pType

  def bytesAddress(): Code[Long] = st.pType.bytesAddress(a)

  override def get: SBinaryPointerCode = new SBinaryPointerCode(st, a)

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(a)

  override def loadLength(): Code[Int] = pt.loadLength(a)

  override def loadBytes(): Code[Array[Byte]] = pt.loadBytes(a)

  override def loadByte(i: Code[Int]): Code[Byte] = Region.loadByte(pt.bytesAddress(a) + i.toL)
}

object SBinaryPointerSettable {
  def apply(sb: SettableBuilder, st: SBinaryPointer, name: String): SBinaryPointerSettable =
    new SBinaryPointerSettable(st, sb.newSettable[Long](name))
}

final class SBinaryPointerSettable(
  st: SBinaryPointer,
  override val a: Settable[Long]
) extends SBinaryPointerValue(st, a) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  override def store(cb: EmitCodeBuilder, pc: SCode): Unit = cb.assign(a, pc.asInstanceOf[SBinaryPointerCode].a)
}

class SBinaryPointerCode(val st: SBinaryPointer, val a: Code[Long]) extends SBinaryCode {
  private val pt: PBinary = st.pType

  def code: Code[_] = a

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadBytes(): Code[Array[Byte]] = pt.loadBytes(a)

  def memoize(cb: EmitCodeBuilder, sb: SettableBuilder, name: String): SBinaryPointerValue = {
    val s = SBinaryPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SBinaryPointerValue =
    memoize(cb, cb.localBuilder, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): SBinaryPointerValue =
    memoize(cb, cb.fieldBuilder, name)
}

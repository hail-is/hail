package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.SBinary
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PBinary, PBinaryCode, PBinaryValue, PCode, PSettable, PType}
import is.hail.utils._


case class SBinaryPointer(pType: PBinary) extends SBinary {
  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SBinaryPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    if (pt == this.pType)
      new SBinaryPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.loadCheapPCode(cb, addr), deepCopy = false)
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SBinaryPointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    new SBinaryPointerSettable(this, a)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SBinaryPointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SBinaryPointerCode(this, a)
  }

  def canonicalPType(): PType = pType
}

object SBinaryPointerSettable {
  def apply(sb: SettableBuilder, st: SBinaryPointer, name: String): SBinaryPointerSettable =
    new SBinaryPointerSettable(st, sb.newSettable[Long](name))
}

class SBinaryPointerSettable(val st: SBinaryPointer, val a: Settable[Long]) extends PBinaryValue with PSettable {
  val pt: PBinary = st.pType

  override def bytesAddress(): Code[Long] = st.pType.bytesAddress(a)

  def get: SBinaryPointerCode = new SBinaryPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadBytes(): Code[Array[Byte]] = pt.loadBytes(a)

  def loadByte(i: Code[Int]): Code[Byte] = Region.loadByte(pt.bytesAddress(a) + i.toL)

  def store(cb: EmitCodeBuilder, pc: PCode): Unit = cb.assign(a, pc.asInstanceOf[SBinaryPointerCode].a)
}

class SBinaryPointerCode(val st: SBinaryPointer, val a: Code[Long]) extends PBinaryCode {
  val pt: PBinary = st.pType

  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadBytes(): Code[Array[Byte]] = pt.loadBytes(a)

  def memoize(cb: EmitCodeBuilder, sb: SettableBuilder, name: String): SBinaryPointerSettable = {
    val s = SBinaryPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SBinaryPointerSettable =
    memoize(cb, cb.localBuilder, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): SBinaryPointerSettable =
    memoize(cb, cb.fieldBuilder, name)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = {
    EmitCodeBuilder.scopedVoid(mb) { cb =>
      pt.storeAtAddress(cb, dst, r, this, false)
    }
  }
}

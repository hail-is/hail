package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PBinary, PBinaryCode, PBinaryValue, PCanonicalBinary, PCode, PSettable, PType}
import is.hail.utils._

trait SBinary extends SType

case class SBinaryPointer(pType: PBinary) extends SBinary {
  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    new SBinaryPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    if (pt == this.pType)
      new SBinaryPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.getPointerTo(cb, addr), deepCopy = false)
  }

}

object SBinaryPointerSettable {
  def apply(sb: SettableBuilder, st: SBinaryPointer, name: String): SBinaryPointerSettable =
    new SBinaryPointerSettable(st, sb.newSettable[Long](name))
}

class SBinaryPointerSettable(val st: SBinaryPointer, val a: Settable[Long]) extends PBinaryValue with PSettable {
  val pt: PBinary = st.pType

  def get: SBinaryPointerCode = new SBinaryPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadBytes(): Code[Array[Byte]] = pt.loadBytes(a)

  def loadByte(i: Code[Int]): Code[Byte] = Region.loadByte(pt.bytesAddress(a) + i.toL)

  def store(cb: EmitCodeBuilder, pc: PCode): Unit = cb += a.store(pc.asInstanceOf[SBinaryPointerCode].a)
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

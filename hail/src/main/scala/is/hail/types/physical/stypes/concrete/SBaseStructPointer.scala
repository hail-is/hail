package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SStructSettable}
import is.hail.types.physical.stypes.{SCode, SSettable, SType}
import is.hail.types.physical.{PBaseStruct, PBaseStructCode, PBaseStructValue, PCode, PStructSettable, PType}
import is.hail.utils.FastIndexedSeq


case class SBaseStructPointer(pType: PBaseStruct) extends SBaseStruct {
  def size: Int = pType.size

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SBaseStructPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    if (pt == this.pType)
      new SBaseStructPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.loadCheapPCode(cb, addr), deepCopy = false)
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SBaseStructPointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    new SBaseStructPointerSettable(this, a)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SBaseStructPointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SBaseStructPointerCode(this, a)
  }

  def canonicalPType(): PType = pType

  override val fieldTypes: Array[SType] = pType.types.map(_.sType)
}


object SBaseStructPointerSettable {
  def apply(sb: SettableBuilder, st: SBaseStructPointer, name: String): SBaseStructPointerSettable = {
    new SBaseStructPointerSettable(st, sb.newSettable(name))
  }
}

class SBaseStructPointerSettable(
  val st: SBaseStructPointer,
  val a: Settable[Long]
) extends PStructSettable {
  val pt: PBaseStruct = st.pType

  def get: PBaseStructCode = new SBaseStructPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.loadCheapPCode(cb, pt.loadField(a, fieldIdx)))
  }

  def store(cb: EmitCodeBuilder, pv: PCode): Unit = {
    cb.assign(a, pv.asInstanceOf[SBaseStructPointerCode].a)
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] = {
    pt.isFieldMissing(a, fieldIdx)
  }
}

class SBaseStructPointerCode(val st: SBaseStructPointer, val a: Code[Long]) extends PBaseStructCode {
  val pt: PBaseStruct = st.pType

  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PBaseStructValue = {
    val s = SBaseStructPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.fieldBuilder)
}

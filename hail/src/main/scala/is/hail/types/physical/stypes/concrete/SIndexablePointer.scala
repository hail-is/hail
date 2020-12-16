package is.hail.types.physical.stypes.concrete

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, IntInfo, LineNumber, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.physical.{PCode, PContainer, PIndexableCode, PIndexableValue, PSettable, PType}
import is.hail.utils.FastIndexedSeq


case class SIndexablePointer(pType: PContainer) extends SContainer {
  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SIndexablePointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, IntInfo, LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    if (pt == this.pType)
      new SIndexablePointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.loadCheapPCode(cb, addr), deepCopy = false)
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SIndexablePointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked], length: Settable[Int@unchecked], elementsAddress: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    assert(length.ti == IntInfo)
    assert(elementsAddress.ti == LongInfo)
    new SIndexablePointerSettable(this, a, length, elementsAddress)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SIndexablePointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SIndexablePointerCode(this, a)
  }
}


class SIndexablePointerCode(val st: SIndexablePointer, val a: Code[Long]) extends PIndexableCode {
  val pt: PContainer = st.pType

  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def loadLength()(implicit line: LineNumber): Code[Int] = pt.loadLength(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PIndexableValue = {
    val s = SIndexablePointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PIndexableValue =
    memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PIndexableValue =
    memoize(cb, name, cb.fieldBuilder)
}

object SIndexablePointerSettable {
  def apply(sb: SettableBuilder, st: SIndexablePointer, name: String): SIndexablePointerSettable = {
    new SIndexablePointerSettable(st,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Int](s"${ name }_length"),
      sb.newSettable[Long](s"${ name }_elems_addr"))
  }
}

class SIndexablePointerSettable(
  val st: SIndexablePointer,
  val a: Settable[Long],
  val length: Settable[Int],
  val elementsAddress: Settable[Long]
) extends PIndexableValue with PSettable {
  val pt: PContainer = st.pType

  def get(implicit line: LineNumber): PIndexableCode = new SIndexablePointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, length, elementsAddress)

  def loadLength(): Value[Int] = length

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    implicit val line = cb.lineNumber
    val iv = cb.newLocal("pcindval_i", i)
    IEmitCode(cb,
      isElementMissing(iv),
      pt.elementType.loadCheapPCode(cb, pt.loadElement(a, length, iv))) // FIXME loadElement should take elementsAddress
  }

  def isElementMissing(i: Code[Int])(implicit line: LineNumber): Code[Boolean] = pt.isElementMissing(a, i)

  def store(cb: EmitCodeBuilder, pc: PCode): Unit = {
    implicit val line = cb.lineNumber
    cb.assign(a, pc.asInstanceOf[SIndexablePointerCode].a)
    cb.assign(length, pt.loadLength(a))
    cb.assign(elementsAddress, pt.firstElementOffset(a, length))
  }
}

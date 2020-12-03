package is.hail.types.physical.stypes.concrete

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.SString
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PBinaryCode, PCanonicalString, PCode, PSettable, PString, PStringCode, PStringValue, PType, PValue}
import is.hail.utils.FastIndexedSeq


case class SStringPointer(pType: PString) extends SString {
  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SStringPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case _: PCanonicalString =>
        new SStringPointerCode(this, addr)
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SStringPointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    new SStringPointerSettable(this, a)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SStringPointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SStringPointerCode(this, a)
  }
}


class SStringPointerCode(val st: SStringPointer, val a: Code[Long]) extends PStringCode {
  override def pt: PString = st.pType

  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadString(): Code[String] = pt.loadString(a)

  def asBytes(): PBinaryCode = new SBinaryPointerCode(SBinaryPointer(pt.fundamentalType), a)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PValue = {
    val s = new SStringPointerSettable(st, sb.newSettable[Long]("sstringpointer_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PValue = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = memoizeWithBuilder(cb, name, cb.fieldBuilder)
}

object SStringPointerSettable {
  def apply(sb: SettableBuilder, st: SStringPointer, name: String): SStringPointerSettable = {
    new SStringPointerSettable(st,
      sb.newSettable[Long](s"${ name }_a"))
  }
}

class SStringPointerSettable(val st: SStringPointer, val a: Settable[Long]) extends PStringValue with PSettable {
  val pt: PString = st.pType

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def get: PCode = new SStringPointerCode(st, a.load())

  def store(cb: EmitCodeBuilder, v: PCode): Unit = {
    cb.assign(a, v.asInstanceOf[SStringPointerCode].a)
  }
}
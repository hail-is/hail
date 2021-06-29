package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.{SBinaryCode, SString, SStringCode, SStringValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalString, PString, PType}
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq


case class SStringPointer(pType: PString) extends SString {
  require(!pType.required)

  lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = this

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SStringPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

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

  def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringPointerCode = {
    new SStringPointerCode(this, pType.allocateAndStoreString(cb.emb, r, s))
  }

  override def canonicalPType(): PType = pType
}


class SStringPointerCode(val st: SStringPointer, val a: Code[Long]) extends SStringCode {
  val pt: PString = st.pType

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadString(): Code[String] = pt.loadString(a)

  def toBytes(): SBinaryPointerCode = new SBinaryPointerCode(SBinaryPointer(pt.binaryRepresentation), a)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SValue = {
    val s = new SStringPointerSettable(st, sb.newSettable[Long]("sstringpointer_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SValue = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SValue = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def binaryRepr: SBinaryPointerCode = new SBinaryPointerCode(SBinaryPointer(st.pType.binaryRepresentation), a)
}

object SStringPointerSettable {
  def apply(sb: SettableBuilder, st: SStringPointer, name: String): SStringPointerSettable = {
    new SStringPointerSettable(st,
      sb.newSettable[Long](s"${ name }_a"))
  }
}

class SStringPointerSettable(val st: SStringPointer, val a: Settable[Long]) extends SStringValue with SSettable {
  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def get: SStringPointerCode = new SStringPointerCode(st, a.load())

  def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    cb.assign(a, v.asInstanceOf[SStringPointerCode].a)
  }

  def binaryRepr(): SBinaryPointerSettable = new SBinaryPointerSettable(SBinaryPointer(st.pType.binaryRepresentation), a)
}

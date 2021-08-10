package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SBaseStructValue, SStructSettable}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.{PBaseStruct, PType}
import is.hail.types.virtual.{TBaseStruct, Type}
import is.hail.utils.FastIndexedSeq


case class SBaseStructPointer(pType: PBaseStruct) extends SBaseStruct {
  require(!pType.required)
  def size: Int = pType.size

  lazy val virtualType: TBaseStruct = pType.virtualType.asInstanceOf[TBaseStruct]

  override def castRename(t: Type): SType = SBaseStructPointer(pType.deepRename(t).asInstanceOf[PBaseStruct])

  override def fieldIdx(fieldName: String): Int = pType.fieldIdx(fieldName)

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SBaseStructPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

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

  override val fieldTypes: IndexedSeq[SType] = pType.types.map(_.sType)
  override val fieldEmitTypes: IndexedSeq[EmitType] = pType.types.map(t => EmitType(t.sType, t.required))
}


object SBaseStructPointerSettable {
  def apply(sb: SettableBuilder, st: SBaseStructPointer, name: String): SBaseStructPointerSettable = {
    new SBaseStructPointerSettable(st, sb.newSettable(name))
  }
}

class SBaseStructPointerSettable(
  val st: SBaseStructPointer,
  val a: Settable[Long]
) extends SStructSettable {
  val pt: PBaseStruct = st.pType

  def get: SBaseStructCode = new SBaseStructPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.loadCheapSCode(cb, pt.loadField(a, fieldIdx)))
  }

  def store(cb: EmitCodeBuilder, pv: SCode): Unit = {
    cb.assign(a, pv.asInstanceOf[SBaseStructPointerCode].a)
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] = {
    pt.isFieldMissing(a, fieldIdx)
  }
}

class SBaseStructPointerCode(val st: SBaseStructPointer, val a: Code[Long]) extends SBaseStructCode {
  val pt: PBaseStruct = st.pType

  def code: Code[_] = a

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SBaseStructValue = {
    val s = SBaseStructPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue = memoize(cb, name, cb.fieldBuilder)
}

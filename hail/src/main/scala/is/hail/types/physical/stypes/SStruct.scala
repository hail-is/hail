package is.hail.types.physical.stypes
import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, IntInfo, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.physical.{PBaseStruct, PBaseStructCode, PBaseStructValue, PCanonicalBaseStruct, PCode, PSettable, PStruct, PType}
import is.hail.utils.FastIndexedSeq

trait SStruct extends SType

trait SStructSettable extends PBaseStructValue with PSettable

case class SBaseStructPointer(pType: PBaseStruct) extends SStruct {
  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SBaseStructPointer(pt2) if pt2.equalModuloRequired(pType) && !deepCopy =>
        value
      case _ =>
        new SBaseStructPointerCode(this, pType.store(cb, region, value, deepCopy))
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, IntInfo, LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    if (pt == this.pType)
      new SBaseStructPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.getPointerTo(cb, addr), deepCopy = false)
  }
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

  def get: PBaseStructCode = new SBaseStructPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.getPointerTo(cb, pt.loadField(a, fieldIdx)))
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

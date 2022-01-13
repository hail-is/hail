package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalLocus, PType}
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq
import is.hail.variant.{Locus, ReferenceGenome}


final case class SCanonicalLocusPointer(pType: PCanonicalLocus) extends SLocus {
  require(!pType.required)

  override def contigType: SString = pType.contigType.sType

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = this

  override def rg: ReferenceGenome = pType.rg

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value match {
      case value: SLocusValue =>
        new SCanonicalLocusPointerValue(this, pType.store(cb, region, value, deepCopy), value.contigLong(cb), value.position(cb))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, LongInfo, IntInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SCanonicalLocusPointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked], contig: Settable[Long@unchecked], position: Settable[Int@unchecked]) = settables
    assert(a.ti == LongInfo)
    assert(contig.ti == LongInfo)
    assert(position.ti == IntInfo)
    new SCanonicalLocusPointerSettable(this, a, contig, position)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SCanonicalLocusPointerValue = {
    val IndexedSeq(a: Value[Long@unchecked], contig: Value[Long@unchecked], position: Value[Int@unchecked]) = values
    assert(a.ti == LongInfo)
    assert(contig.ti == LongInfo)
    assert(position.ti == IntInfo)
    new SCanonicalLocusPointerValue(this, a, contig, position)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SCanonicalLocusPointer(pType.copiedType.asInstanceOf[PCanonicalLocus])

  override def containsPointers: Boolean = pType.containsPointers
}

class SCanonicalLocusPointerValue(
  val st: SCanonicalLocusPointer,
  val a: Value[Long],
  val _contig: Value[Long],
  val _position: Value[Int]
) extends SLocusValue {
  val pt: PCanonicalLocus = st.pType

  override def get = new SCanonicalLocusPointerCode(st, a)

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(a, _contig, _position)

  override def contig(cb: EmitCodeBuilder): SStringValue = {
    pt.contigType.loadCheapSCode(cb, _contig).asString
  }

  override def contigLong(cb: EmitCodeBuilder): Value[Long] = _contig

  override def position(cb: EmitCodeBuilder): Value[Int] = _position

  override def structRepr(cb: EmitCodeBuilder): SBaseStructValue = new SBaseStructPointerValue(
    SBaseStructPointer(st.pType.representation), a)
}

object SCanonicalLocusPointerSettable {
  def apply(sb: SettableBuilder, st: SCanonicalLocusPointer, name: String): SCanonicalLocusPointerSettable = {
    new SCanonicalLocusPointerSettable(st,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Long](s"${ name }_contig"),
      sb.newSettable[Int](s"${ name }_position"))
  }
}

final class SCanonicalLocusPointerSettable(
  st: SCanonicalLocusPointer,
  override val a: Settable[Long],
  _contig: Settable[Long],
  override val _position: Settable[Int]
) extends SCanonicalLocusPointerValue(st, a, _contig, _position) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, _contig, _position)

  override def store(cb: EmitCodeBuilder, pc: SCode): Unit = {
    cb.assign(a, pc.asInstanceOf[SCanonicalLocusPointerCode].a)
    cb.assign(_contig, pt.contigAddr(a))
    cb.assign(_position, pt.position(a))
  }

  override def structRepr(cb: EmitCodeBuilder): SBaseStructPointerSettable = new SBaseStructPointerSettable(
    SBaseStructPointer(st.pType.representation), a)
}

class SCanonicalLocusPointerCode(val st: SCanonicalLocusPointer, val a: Code[Long]) extends SLocusCode {
  val pt: PCanonicalLocus = st.pType

  def code: Code[_] = a

  def position(cb: EmitCodeBuilder): Code[Int] = pt.position(a)

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus] = {
    val loc = memoize(cb, "get_locus_code_memo")
    Code.newInstance[Locus, String, Int](loc.contig(cb).asString.loadString(cb), loc.position(cb))
  }

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SCanonicalLocusPointerSettable = {
    val s = SCanonicalLocusPointerSettable(sb, st, name)
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SCanonicalLocusPointerSettable = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SCanonicalLocusPointerSettable = memoize(cb, name, cb.fieldBuilder)

  def structRepr(cb: EmitCodeBuilder): SBaseStructCode = new SBaseStructPointerCode(SBaseStructPointer(st.pType.representation), a)
}

package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.SLocus
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PCanonicalLocus, PCode, PLocusCode, PLocusValue, PSettable, PStringCode, PType}
import is.hail.utils.FastIndexedSeq
import is.hail.variant.{Locus, ReferenceGenome}


case class SCanonicalLocusPointer(pType: PCanonicalLocus) extends SLocus {
  override def rg: ReferenceGenome = pType.rg

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SCanonicalLocusPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case PCanonicalLocus(_, _) =>
        new SCanonicalLocusPointerCode(this, addr)
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SCanonicalLocusPointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked], contig: Settable[Long@unchecked], position: Settable[Int@unchecked]) = settables
    assert(a.ti == LongInfo)
    assert(contig.ti == LongInfo)
    assert(position.ti == IntInfo)
    new SCanonicalLocusPointerSettable(this, a, contig, position)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SCanonicalLocusPointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SCanonicalLocusPointerCode(this, a)
  }

  def canonicalPType(): PType = pType
}


object SCanonicalLocusPointerSettable {
  def apply(sb: SettableBuilder, st: SCanonicalLocusPointer, name: String): SCanonicalLocusPointerSettable = {
    new SCanonicalLocusPointerSettable(st,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Long](s"${ name }_contig"),
      sb.newSettable[Int](s"${ name }_position"))
  }
}

class SCanonicalLocusPointerSettable(
  val st: SCanonicalLocusPointer,
  val a: Settable[Long],
  _contig: Settable[Long],
  val _position: Settable[Int]
) extends PLocusValue with PSettable {
  val pt: PCanonicalLocus = st.pType

  def get = new SCanonicalLocusPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, _contig, _position)

  def store(cb: EmitCodeBuilder, pc: PCode): Unit = {
    cb.assign(a, pc.asInstanceOf[SCanonicalLocusPointerCode].a)
    cb.assign(_contig, pt.contigAddr(a))
    cb.assign(_position, pt.position(a))
  }

  def contig(cb: EmitCodeBuilder): PStringCode = {
    pt.contigType.loadCheapPCode(cb, _contig).asString
  }

  def position(cb: EmitCodeBuilder): Code[Int] = _position
}

class SCanonicalLocusPointerCode(val st: SCanonicalLocusPointer, val a: Code[Long]) extends PLocusCode {
  val pt: PCanonicalLocus = st.pType

  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def contig(cb: EmitCodeBuilder): PStringCode = pt.contigType.loadCheapPCode(cb, pt.contigAddr(a)).asString

  def position(cb: EmitCodeBuilder): Code[Int] = pt.position(a)

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus] = {
    val loc = memoize(cb, "get_locus_code_memo")
    Code.newInstance[Locus, String, Int](loc.contig(cb).asString.loadString(), loc.position(cb))
  }

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SCanonicalLocusPointerSettable = {
    val s = SCanonicalLocusPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SCanonicalLocusPointerSettable = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SCanonicalLocusPointerSettable = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}

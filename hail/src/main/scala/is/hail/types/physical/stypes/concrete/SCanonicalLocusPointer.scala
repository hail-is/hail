package is.hail.types.physical.stypes.concrete

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.SLocus
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PCanonicalLocus, PCode, PLocusCode, PLocusValue, PSettable, PStringCode, PType}
import is.hail.utils.FastIndexedSeq
import is.hail.variant.Locus


case class SCanonicalLocusPointer(pType: PCanonicalLocus) extends SLocus {
  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean)(implicit line: LineNumber): SCode = {
    new SCanonicalLocusPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long])(implicit line: LineNumber): SCode = {
    pt match {
      case PCanonicalLocus(_, _) =>
        new SCanonicalLocusPointerCode(this, addr)
    }
  }
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

  def get(implicit line: LineNumber) = new SCanonicalLocusPointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, _contig, _position)

  def store(cb: EmitCodeBuilder, pc: PCode)(implicit line: LineNumber): Unit = {
    cb.assign(a, pc.asInstanceOf[SCanonicalLocusPointerCode].a)
    cb.assign(_contig, pt.contigAddr(a))
    cb.assign(_position, pt.position(a))
  }

  def contig(cb: EmitCodeBuilder)(implicit line: LineNumber): PStringCode = {
    pt.contigType.loadCheapPCode(cb, _contig).asString
  }

  def position(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Int] = _position
}

class SCanonicalLocusPointerCode(val st: SCanonicalLocusPointer, val a: Code[Long]) extends PLocusCode {
  val pt: PCanonicalLocus = st.pType

  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def contig(cb: EmitCodeBuilder)(implicit line: LineNumber): PStringCode =
    pt.contigType.loadCheapPCode(cb, pt.contigAddr(a)).asString

  def position(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Int] =
    pt.position(a)

  def getLocusObj(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Locus] = {
    val loc = memoize(cb, "get_locus_code_memo")
    Code.newInstance[Locus, String, Int](loc.contig(cb).asString.loadString(), loc.position(cb))
  }

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder)(implicit line: LineNumber): SCanonicalLocusPointerSettable = {
    val s = SCanonicalLocusPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): SCanonicalLocusPointerSettable =
    memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): SCanonicalLocusPointerSettable =
    memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long])(implicit line: LineNumber): Code[Unit] =
    Region.storeAddress(dst, a)
}

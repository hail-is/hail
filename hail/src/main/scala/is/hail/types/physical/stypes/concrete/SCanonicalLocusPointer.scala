package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitSettable, EmitValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Settable, SInt32Value}
import is.hail.types.physical.stypes.{EmitType, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalLocus, PType}
import is.hail.types.virtual.{TLocus, Type}
import is.hail.utils.FastSeq


final case class SCanonicalLocusPointer(pType: PCanonicalLocus) extends SLocus {
  require(!pType.required)

  override def contigType: SString = pType.contigType.sType

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = this

  override def rg: String = pType.rg

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value match {
      case value: SLocusValue =>
        val locusCopy = pType.store(cb, region, value, deepCopy)
        val contigCopy = cb.memoize(pType.contigAddr(locusCopy))
        new SCanonicalLocusPointerValue(this, locusCopy, contigCopy, value.position(cb))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo, LongInfo, IntInfo)

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

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a, _contig, _position)

  override def contig(cb: EmitCodeBuilder): SStringValue = {
    pt.contigType.loadCheapSCode(cb, _contig).asString
  }

  override def contigIdx(cb: EmitCodeBuilder): Value[Int] = {
    cb.memoize(cb.emb.getReferenceGenome(st.rg).invoke[String, Int]("getContigIdx", contig(cb).loadString(cb)))
  }

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
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a, _contig, _position)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SCanonicalLocusPointerValue =>
      cb.assign(a, v.a)
      cb.assign(_contig, v._contig)
      cb.assign(_position, v._position)
  }

  override def structRepr(cb: EmitCodeBuilder): SBaseStructPointerSettable = new SBaseStructPointerSettable(
    SBaseStructPointer(st.pType.representation), a)
}


final case class SCompactLocus(rg: String) extends SLocus {
  lazy val virtualType: TLocus = TLocus(rg)
  lazy val pType: PCanonicalLocus = PCanonicalLocus(rg)
  override def contigType: SString = SJavaString

  override def containsPointers = false

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value match {
      case value: SLocusValue =>
        val contig = cb.memoize(cb.emb.getReferenceGenome(rg).invoke[String, Int]("getContigIndex", value.contig(cb).loadString(cb)))
        new SCompactLocusValue(this, contig, value.position(cb))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(IntInfo, IntInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SCompactLocusSettable = {
    val IndexedSeq(contig: Settable[Int@unchecked], position: Settable[Int@unchecked]) = settables
    assert(contig.ti == IntInfo)
    assert(position.ti == IntInfo)
    new SCompactLocusSettable(this, contig, position)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SCompactLocusValue = {
    val IndexedSeq(contig: Value[Int@unchecked], position: Value[Int@unchecked]) = values
    assert(contig.ti == IntInfo)
    assert(position.ti == IntInfo)
    new SCompactLocusValue(this, contig, position)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = this

  override def castRename(t: Type): SType = this
}

class SCompactLocusValue(
  val st: SCompactLocus,
  val _contig: Value[Int],
  val _position: Value[Int]
) extends SLocusValue {
  override def contig(cb: EmitCodeBuilder): SStringValue = {
    val contig = cb.memoize(cb.emb.getReferenceGenome(st.rg).invoke[Int, String]("getContig", _contig))
    new SJavaStringValue(contig)
  }

  override def contigIdx(cb: EmitCodeBuilder): Value[Int] = _contig

  override def position(cb: EmitCodeBuilder): Value[Int] = _position

  override def structRepr(cb: EmitCodeBuilder): SBaseStructValue = new SStackStructValue(
    SStackStruct(
      st.virtualType.representation,
      FastSeq(EmitType(SJavaString, true), EmitType(SInt32, true))),
    FastSeq(
      EmitValue.present(contig(cb)),
      EmitValue.present(new SInt32Value(_position))))

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq(_contig, _position)
}

final class SCompactLocusSettable(
  st: SCompactLocus,
  override val _contig: Settable[Int],
  override val _position: Settable[Int]
) extends SCompactLocusValue(st, _contig, _position) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(_contig, _position)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SCompactLocusValue =>
      cb.assign(_contig, v._contig)
      cb.assign(_position, v._position)
  }

  override def structRepr(cb: EmitCodeBuilder): SStackStructSettable = new SStackStructSettable(
    SStackStruct(
      st.virtualType.representation,
      FastSeq(EmitType(SInt32, true), EmitType(SInt32, true))),
    FastSeq(EmitSettable.present(new SInt32Settable(_contig)),
            EmitSettable.present(new SInt32Settable(_position))))
}

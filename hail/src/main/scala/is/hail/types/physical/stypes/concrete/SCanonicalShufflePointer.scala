package is.hail.types.physical.stypes.concrete

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, IntInfo, LineNumber, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.services.shuffler.Wire
import is.hail.types.physical.stypes.interfaces.SShuffle
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PCanonicalShuffle, PCode, PSettable, PShuffle, PShuffleCode, PShuffleValue, PType}
import is.hail.utils.FastIndexedSeq

case class SCanonicalShufflePointer(pType: PCanonicalShuffle) extends SShuffle {

  lazy val binarySType = SBinaryPointer(pType.representation)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SCanonicalShufflePointerCode(this, pType.representation.loadCheapPCode(cb, pType.store(cb, region, value, deepCopy)))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, IntInfo, IntInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case t: PCanonicalShuffle =>
        assert(t.equalModuloRequired(this.pType))
        new SCanonicalShufflePointerCode(this, t.loadBinary(cb, addr))
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SCanonicalShufflePointerSettable = {
    new SCanonicalShufflePointerSettable(this, binarySType.fromSettables(settables))
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SCanonicalShufflePointerCode = {
    new SCanonicalShufflePointerCode(this, binarySType.fromCodes(codes))
  }
}

object SCanonicalShufflePointerSettable {
  def apply(sb: SettableBuilder, st: SCanonicalShufflePointer, name: String): SCanonicalShufflePointerSettable =
    new SCanonicalShufflePointerSettable(st, SBinaryPointerSettable(sb, SBinaryPointer(st.pType.representation), name))

  def fromArrayBytes(cb: EmitCodeBuilder, region: Value[Region], pt: PCanonicalShuffle, bytes: Code[Array[Byte]]): SCanonicalShufflePointerSettable = {
    implicit val line = cb.lineNumber
    val off = cb.newField[Long](
      "PCanonicalShuffleSettableOff",
      pt.representation.allocate(region, Wire.ID_SIZE))
    cb.append(pt.representation.store(off, bytes))
    SCanonicalShufflePointer(pt).loadFrom(cb, region, pt, off).memoize(cb, "scanonicalshuffle_fromarraybytes").asInstanceOf[SCanonicalShufflePointerSettable]
  }
}

class SCanonicalShufflePointerSettable(val st: SCanonicalShufflePointer, shuffle: SBinaryPointerSettable) extends PShuffleValue with PSettable {
  val pt: PShuffle = st.pType

  def get(implicit line: LineNumber): PShuffleCode =
    new SCanonicalShufflePointerCode(st, shuffle.get)

  def settableTuple(): IndexedSeq[Settable[_]] = shuffle.settableTuple()

  def loadLength()(implicit line: LineNumber): Code[Int] = shuffle.loadLength()

  def loadBytes()(implicit line: LineNumber): Code[Array[Byte]] = shuffle.loadBytes()

  def store(cb: EmitCodeBuilder, pc: PCode): Unit =
    shuffle.store(cb, pc.asInstanceOf[SCanonicalShufflePointerCode].shuffle)
}

class SCanonicalShufflePointerCode(val st: SCanonicalShufflePointer, val shuffle: SBinaryPointerCode) extends PShuffleCode {
  val pt: PShuffle = st.pType

  def code: Code[_] = shuffle.code

  def codeTuple(): IndexedSeq[Code[_]] = shuffle.codeTuple()

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SCanonicalShufflePointerSettable = {
    val s = SCanonicalShufflePointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SCanonicalShufflePointerSettable =
    memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SCanonicalShufflePointerSettable =
    memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long])(implicit line: LineNumber): Code[Unit] =
    shuffle.store(mb, r, dst)
}

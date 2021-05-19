package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, IntInfo, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.services.shuffler.Wire
import is.hail.types.physical.stypes.interfaces.{SShuffle, SShuffleCode, SShuffleValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SType}
import is.hail.types.physical.{PCanonicalShuffle, PShuffle, PType}
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq

case class SCanonicalShufflePointer(pType: PCanonicalShuffle) extends SShuffle {
  require(!pType.required)

  lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = this

  lazy val binarySType = SBinaryPointer(pType.representation)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SCanonicalShufflePointerCode(this, pType.representation.loadCheapPCode(cb, pType.store(cb, region, value, deepCopy)))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SCanonicalShufflePointerSettable = {
    new SCanonicalShufflePointerSettable(this, binarySType.fromSettables(settables))
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SCanonicalShufflePointerCode = {
    new SCanonicalShufflePointerCode(this, binarySType.fromCodes(codes))
  }

  def canonicalPType(): PType = pType
}

object SCanonicalShufflePointerSettable {
  def apply(sb: SettableBuilder, st: SCanonicalShufflePointer, name: String): SCanonicalShufflePointerSettable =
    new SCanonicalShufflePointerSettable(st, SBinaryPointerSettable(sb, SBinaryPointer(st.pType.representation), name))

  def fromArrayBytes(cb: EmitCodeBuilder, region: Value[Region], pt: PCanonicalShuffle, bytes: Code[Array[Byte]]): SCanonicalShufflePointerSettable = {
    val off = cb.newField[Long](
      "PCanonicalShuffleSettableOff",
      pt.representation.allocate(region, Wire.ID_SIZE))
    cb.append(pt.representation.store(off, bytes))
    pt.loadCheapPCode(cb, off).memoize(cb, "scanonicalshuffle_fromarraybytes").asInstanceOf[SCanonicalShufflePointerSettable]
  }
}

class SCanonicalShufflePointerSettable(val st: SCanonicalShufflePointer, val shuffle: SBinaryPointerSettable) extends SShuffleValue with SSettable {
  val pt: PCanonicalShuffle = st.pType

  def get: SShuffleCode = new SCanonicalShufflePointerCode(st, shuffle.get)

  def settableTuple(): IndexedSeq[Settable[_]] = shuffle.settableTuple()

  def loadLength(): Code[Int] = shuffle.loadLength()

  def loadBytes(): Code[Array[Byte]] = shuffle.loadBytes()

  def store(cb: EmitCodeBuilder, pc: SCode): Unit = shuffle.store(cb, pc.asInstanceOf[SCanonicalShufflePointerCode].shuffle)

  def storeFromBytes(cb: EmitCodeBuilder, region: Value[Region], bytes: Value[Array[Byte]]): Unit = {
    val addr = cb.newLocal[Long]("bytesAddr", st.pType.representation.allocate(region, bytes.length()))
    cb += st.pType.representation.store(addr, bytes)
    shuffle.store(cb, st.pType.representation.loadCheapPCode(cb, addr))
  }
}

class SCanonicalShufflePointerCode(val st: SCanonicalShufflePointer, val shuffle: SBinaryPointerCode) extends SShuffleCode {
  val pt: PShuffle = st.pType

  def code: Code[_] = shuffle.code

  def codeTuple(): IndexedSeq[Code[_]] = shuffle.codeTuple()

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SCanonicalShufflePointerSettable = {
    val s = SCanonicalShufflePointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SCanonicalShufflePointerSettable = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SCanonicalShufflePointerSettable = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = shuffle.store(mb, r, dst)

  def binaryRepr: SBinaryPointerCode = shuffle
}

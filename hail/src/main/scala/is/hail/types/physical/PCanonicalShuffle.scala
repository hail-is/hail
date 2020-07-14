package is.hail.types.physical

import is.hail.services.shuffler.Wire
import is.hail.types.virtual._
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.utils._

final case class PCanonicalShuffle(
  val tShuffle: TShuffle,
  required: Boolean = false
) extends PShuffle {
  def _asIdent = "shuffle"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCShuffle")

  val representation: PCanonicalBinary = PCanonicalBinary(required)

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    representation.codeOrdering(mb)
  }

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalShuffle(tShuffle, required)

  def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()
}

object PCanonicalShuffleSettable {
  def apply(sb: SettableBuilder, pt: PCanonicalShuffle, name: String): PCanonicalShuffleSettable =
    new PCanonicalShuffleSettable(pt, PCanonicalBinarySettable(sb, pt.representation, name))

  def fromArrayBytes(cb: EmitCodeBuilder, region: Value[Region], pt: PCanonicalShuffle, bytes: Code[Array[Byte]]) = {
    val off = cb.newField[Long](
      "PCanonicalShuffleSettableOff",
      pt.representation.allocate(region, Wire.ID_SIZE))
    cb.append(pt.representation.store(off, bytes))
    new PCanonicalShuffleSettable(pt, new PCanonicalBinarySettable(pt.representation, off))
  }
}

class PCanonicalShuffleSettable(val pt: PCanonicalShuffle, shuffle: PCanonicalBinarySettable) extends PShuffleValue with PSettable {
  def this(pt: PCanonicalShuffle, a: Settable[Long]) =
    this(pt, new PCanonicalBinarySettable(pt.representation, a))

  def get: PShuffleCode = new PCanonicalShuffleCode(pt, shuffle.get)

  def settableTuple(): IndexedSeq[Settable[_]] = shuffle.settableTuple()

  def loadLength(): Code[Int] = shuffle.loadLength()

  def loadBytes(): Code[Array[Byte]] = shuffle.loadBytes()

  def store(pc: PCode): Code[Unit] = shuffle.store(pc.asInstanceOf[PCanonicalShuffleCode].shuffle)
}

class PCanonicalShuffleCode(val pt: PCanonicalShuffle, val shuffle: PCanonicalBinaryCode) extends PShuffleCode {
  def code: Code[_] = shuffle.code

  def codeTuple(): IndexedSeq[Code[_]] = shuffle.codeTuple()

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PCanonicalShuffleSettable = {
    val s = PCanonicalShuffleSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PCanonicalShuffleSettable = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PCanonicalShuffleSettable = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = shuffle.store(mb, r, dst)
}

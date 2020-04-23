package is.hail.types.physical

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
}

object PCanonicalShuffleSettable {
  def apply(sb: SettableBuilder, pt: PCanonicalShuffle, name: String): PCanonicalShuffleSettable =
    new PCanonicalShuffleSettable(pt, sb.newSettable[Long](s"${ name }_shuffle"))
}

class PCanonicalShuffleSettable(val pt: PCanonicalShuffle, shuffle: Settable[Long]) extends PShuffleValue with PSettable {
  private[this] val representation = new PCanonicalBinarySettable(
    pt.representation, shuffle)

  def get: PShuffleCode = new PCanonicalShuffleCode(pt, shuffle)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(shuffle)

  def loadLength(): Code[Int] = representation.loadLength()

  def loadBytes(): Code[Array[Byte]] = representation.loadBytes()

  def loadByte(i: Code[Int]): Code[Byte] = representation.loadByte(i)

  def store(pc: PCode): Code[Unit] = shuffle.store(pc.asInstanceOf[PCanonicalShuffleCode].shuffle)
}

class PCanonicalShuffleCode(val pt: PCanonicalShuffle, val shuffle: Code[Long]) extends PShuffleCode {
  def code: Code[_] = shuffle

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(shuffle)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PCanonicalShuffleSettable = {
    val s = PCanonicalShuffleSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PCanonicalShuffleSettable = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PCanonicalShuffleSettable = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeLong(dst, shuffle)
}

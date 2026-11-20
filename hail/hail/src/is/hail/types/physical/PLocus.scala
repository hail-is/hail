package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.SLocusValue
import is.hail.types.virtual.TLocus
import is.hail.utils.fatal
import is.hail.variant.{Locus, ReferenceGenome}

object PLocus {

  /** Packs a locus on `rg` into a long as `(contigIdx << 32) | position`. Fails on a negative
    * position: its sign bits would overwrite the contig index. Every site that packs a locus calls
    * this or the staged `pack`, so none of them skips the check. Packed loci that are read rather
    * than packed, such as those decoded from Arrow or the varint encoding, are not checked.
    */
  def pack(rg: ReferenceGenome, contigIdx: Int, position: Int): Long = {
    if (position < 0) fatal(negativePositionMessage(rg, contigIdx, position))
    (contigIdx.toLong << 32) | position.toLong
  }

  /** The staged [[pack]]. */
  def pack(cb: EmitCodeBuilder, rg: ReferenceGenome, contigIdx: Code[Int], position: Code[Int])
    : Value[Long] = {
    val idx = cb.memoize(contigIdx)
    val pos = cb.memoize(position)
    cb.if_(
      pos < 0,
      cb._fatal(Code.invokeScalaObject3[ReferenceGenome, Int, Int, String](
        PLocus.getClass,
        "negativePositionMessage",
        cb.emb.getReferenceGenome(rg),
        idx,
        pos,
      )),
    )
    cb.memoize((idx.toL << 32) | pos.toL)
  }

  def negativePositionMessage(rg: ReferenceGenome, contigIdx: Int, position: Int): String =
    s"Invalid locus '${rg.getContig(contigIdx)}:$position' for reference genome '${rg.name}': " +
      "a locus position cannot be negative."

  // Volatile and global, not thread-local, so that Spark tasks running in the test JVM see the
  // same choice as the driver.
  @volatile private[this] var packedIsCanonical: Boolean = false

  /** The locus type that a locus gets unless something else chooses one, such as the canonical
    * PType of a [[TLocus]]. For now that is a [[PCanonicalLocus]]; packed loci become canonical
    * once every locus producer goes through here.
    */
  def canonical(rg: ReferenceGenome, required: Boolean = false): PLocus =
    if (packedIsCanonical) PPackedLocus(rg, required) else PCanonicalLocus(rg, required)

  /** The type of a locus field that may have no reference genome, such as one read from a file: the
    * canonical locus type on `rg`, or a struct of contig and position without one.
    */
  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): PType = rg match {
    case Some(rg) => canonical(rg, required)
    case None => PCanonicalLocus.representation(required)
  }

  /** Runs `f` with packed (or struct) loci as the canonical locus representation. For tests only:
    * code compiled inside `f` must not outlive it.
    */
  private[hail] def withPackedCanonical[A](packed: Boolean)(f: => A): A = {
    val saved = packedIsCanonical
    packedIsCanonical = packed
    try f
    finally packedIsCanonical = saved
  }
}

abstract class PLocus extends PType {
  lazy val virtualType: TLocus = TLocus(rg)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def position(value: Code[Long]): Code[Int]

  def position(value: Long): Int

  def packed(value: Long): Long

  def positionType: PInt32

  def unstagedStoreLocus(
    addr: Long,
    contig: String,
    position: Int,
    region: Region,
  ): Unit

  /** Builds a locus of this type, allocating in `region` if the representation needs it. It does
    * not validate the locus: callers that need validation check it first. A packed locus looks up
    * the contig's index, so it fails on a contig that isn't in `rg`, and it fails on a negative
    * position (see [[PLocus.pack]]). A struct locus accepts either, and fails the same way later if
    * it is packed.
    */
  def constructFromContigAndPosition(
    cb: EmitCodeBuilder,
    region: Value[Region],
    contig: Code[String],
    position: Code[Int],
  ): SLocusValue

  def constructFromLocus(cb: EmitCodeBuilder, region: Value[Region], locus: Code[Locus])
    : SLocusValue = {
    val l = cb.memoize(locus)
    constructFromContigAndPosition(
      cb,
      region,
      l.invoke[String]("contig"),
      l.invoke[Int]("position"),
    )
  }

  override def unsafeOrdering(rightType: PType): UnsafeOrdering =
    rightType match {
      case right: PLocus if right.getClass != getClass =>
        require(virtualType == right.virtualType, s"$this, $right")
        new UnsafeOrdering {
          override def compare(o1: Long, o2: Long): Int =
            java.lang.Long.compare(packed(o1), right.packed(o2))
        }
      case _ =>
        super.unsafeOrdering(rightType)
    }
}

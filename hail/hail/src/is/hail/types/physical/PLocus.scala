package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.SLocusValue
import is.hail.types.virtual.TLocus
import is.hail.variant.{Locus, ReferenceGenome}

object PLocus {
  def pack(contigIdx: Int, position: Int): Long =
    (contigIdx.toLong << 32) | position.toLong

  // Volatile and global, not thread-local, so that Spark tasks running in the test JVM see the
  // same choice as the driver.
  @volatile private[this] var packedIsCanonical: Boolean = false

  /** The locus type that a locus gets unless something else chooses one, such as the canonical
    * PType of a [[TLocus]]. For now that is a [[PCanonicalLocus]]; packed loci become canonical
    * once every locus producer goes through here.
    */
  def canonical(rg: ReferenceGenome, required: Boolean = false): PLocus =
    if (packedIsCanonical) PPackedLocus(rg, required) else PCanonicalLocus(rg, required)

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
    * the contig's index, so it fails on a contig that isn't in `rg`. `position` must not be
    * negative: a packed locus keeps it in its low 32 bits, where a negative position would
    * overwrite the contig index.
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

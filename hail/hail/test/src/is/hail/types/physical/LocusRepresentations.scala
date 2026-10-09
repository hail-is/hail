package is.hail.types.physical

import is.hail.annotations.Annotation
import is.hail.backend.ExecuteContext
import is.hail.scalacheck._
import is.hail.variant.BuiltinLoci.builtin
import is.hail.variant.ReferenceGenome

import scala.collection.immutable.ArraySeq

import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen

/** Helpers for testing loci in either representation, against a builtin and a custom reference
  * genome.
  */
object LocusRepresentations {

  val customReference: ReferenceGenome =
    ReferenceGenome(
      "packed_locus_custom",
      ArraySeq("c", "a", "b"),
      Map("a" -> 100, "b" -> 2000, "c" -> 30000),
    )

  val references: IndexedSeq[ReferenceGenome] =
    ArraySeq(builtin(ReferenceGenome.GRCh38), customReference)

  /** A copy of `rg` that is a distinct object, as when two sessions define the same layout. */
  def twin(rg: ReferenceGenome): ReferenceGenome = {
    val copy = rg.copy()
    assert(copy ne rg)
    copy
  }

  /** Runs `f` in a session that has `rg` registered. Only code that resolves reference genomes by
    * name, such as parsing or reading a dataset, needs this: locus types hold their reference
    * genome.
    */
  def withReference[A](rg: ReferenceGenome)(f: ExecuteContext => A)(implicit ctx: ExecuteContext)
    : A =
    ctx.local(references = ctx.references.withReferences(Seq(rg)))(f)

  def mapLoci(pt: PType)(f: PLocus => PLocus): PType = pt match {
    case l: PLocus => f(l)
    case PCanonicalArray(e, r) => PCanonicalArray(mapLoci(e)(f), r)
    case PCanonicalSet(e, r) => PCanonicalSet(mapLoci(e)(f), r)
    case PCanonicalDict(k, v, r) => PCanonicalDict(mapLoci(k)(f), mapLoci(v)(f), r)
    case PCanonicalInterval(p, r) => PCanonicalInterval(mapLoci(p)(f), r)
    case t: PCanonicalStruct =>
      PCanonicalStruct(t.fields.map(fd => fd.copy(typ = mapLoci(fd.typ)(f))), t.required)
    case t: PCanonicalTuple =>
      PCanonicalTuple(t._types.map(fd => fd.copy(typ = mapLoci(fd.typ)(f))), t.required)
    case _ => pt
  }

  /** The loci in `pt`, wherever they are nested. */
  def loci(pt: PType): Seq[PLocus] = {
    val b = Seq.newBuilder[PLocus]
    val _ = mapLoci(pt) { l => b += l; l }
    b.result()
  }

  def toCanonical(pt: PType): PType = mapLoci(pt)(l => PCanonicalLocus(l.rg, l.required))

  def toPacked(pt: PType): PType = mapLoci(pt)(l => PPackedLocus(l.rg, l.required))

  def onReference(pt: PType, rg: ReferenceGenome): PType = mapLoci(pt) {
    case l: PCanonicalLocus => PCanonicalLocus(rg, l.required)
    case l: PPackedLocus => PPackedLocus(rg, l.required)
  }

  /** `pt` in each locus representation, on `rg`. */
  def representations(pt: PType, rg: ReferenceGenome): IndexedSeq[PType] =
    ArraySeq(onReference(toCanonical(pt), rg), onReference(toPacked(pt), rg))

  /** Pairs of `pt` in each locus representation: the first on `rg`, the second on `rg` or on a
    * [[twin]] of it. These exercise conversions between representations, and between equal
    * reference genomes that are distinct objects.
    */
  def representationPairs(pt: PType, rg: ReferenceGenome): IndexedSeq[(PType, PType)] =
    for {
      src <- representations(pt, rg)
      dst <- representations(pt, rg) ++ representations(pt, twin(rg))
    } yield (src, dst)

  /** An arbitrary PType whose loci, of either representation, are on `rg`, with a non-missing
    * value.
    */
  def genPTypeValOn(ctx: ExecuteContext, rg: ReferenceGenome): Gen[(PType, Annotation)] =
    for {
      factor <- beta(1, 8)
      pt <- scale(factor, arbitrary[PType]).map(onReference(_, rg))
      v <- scale(1 - factor, genVal(ctx, pt))
      if v != null
    } yield (pt, v)
}

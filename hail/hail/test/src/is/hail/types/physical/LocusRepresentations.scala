package is.hail.types.physical

import is.hail.annotations.Annotation
import is.hail.backend.ExecuteContext
import is.hail.scalacheck._
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

  def references(implicit ctx: ExecuteContext): IndexedSeq[ReferenceGenome] =
    ArraySeq(ctx.references(ReferenceGenome.GRCh38), customReference)

  def withReference[A](rg: ReferenceGenome)(f: ExecuteContext => A)(implicit ctx: ExecuteContext)
    : A =
    ctx.local(references = ctx.references + (rg.name -> rg))(f)

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

  def toCanonical(pt: PType): PType = mapLoci(pt)(l => PCanonicalLocus(l.rg, l.required))

  def toPacked(pt: PType): PType = mapLoci(pt)(l => PPackedLocus(l.rg, l.required))

  def onReference(pt: PType, rg: String): PType = mapLoci(pt) {
    case l: PCanonicalLocus => PCanonicalLocus(rg, l.required)
    case l: PPackedLocus => PPackedLocus(rg, l.required)
  }

  /** An arbitrary PType whose loci, of either representation, are on `rg`, with a non-missing
    * value. `rg` must be registered in `ctx`.
    */
  def genPTypeValOn(ctx: ExecuteContext, rg: ReferenceGenome): Gen[(PType, Annotation)] =
    for {
      factor <- beta(1, 8)
      pt <- scale(factor, arbitrary[PType]).map(onReference(_, rg.name))
      v <- scale(1 - factor, genVal(ctx, pt))
      if v != null
    } yield (pt, v)
}

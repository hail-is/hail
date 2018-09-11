package is.hail.testUtils

import is.hail.annotations.Annotation
import is.hail.expr.types.{TArray, TLocus, TString, TStruct}
import is.hail.variant._
import org.apache.spark.sql.Row
import org.json4s._

import scala.collection.JavaConverters._

object Variant {
  def apply(contig: String,
    start: Int,
    ref: String,
    alt: String): Variant = {
    Variant(contig, start, ref, Array(AltAllele(ref, alt)))
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String]): Variant = Variant(contig, start, ref, alts.map(alt => AltAllele(ref, alt)))

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String],
    rg: RGBase): Variant = {
    rg.checkLocus(contig, start)
    Variant(contig, start, ref, alts)
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: java.util.ArrayList[String],
    rg: RGBase): Variant = Variant(contig, start, ref, alts.asScala.toArray, rg)

  def fromLocusAlleles(a: Annotation): Variant = {
    val r = a.asInstanceOf[Row]
    val l = r.getAs[Locus](0)
    val alleles = r.getAs[IndexedSeq[String]](1)
    if (l == null || alleles == null)
      null
    else
      Variant(l.contig, l.position, alleles(0), alleles.tail.map(x => AltAllele(alleles(0), x)))
  }
}

case class Variant(contig: String,
  start: Int,
  ref: String,
  altAlleles: IndexedSeq[AltAllele]) {
  require(altAlleles.forall(_.ref == ref))

  /* The position is 1-based. Telomeres are indicated by using positions 0 or N+1, where N is the length of the
       corresponding chromosome or contig. See the VCF spec, v4.2, section 1.4.1. */
  require(start >= 0, s"invalid variant: negative position: `${ this.toString }'")
  require(!ref.isEmpty, s"invalid variant: empty contig: `${ this.toString }'")

  def toLocusAlleles: Row = Row(locus, IndexedSeq(ref) ++ altAlleles.map(_.alt))

  def alleles: IndexedSeq[String] = {
    val a = new Array[String](nAlleles)
    a(0) = ref
    var i = 1
    while (i < a.length) {
      a(i) = altAlleles(i - 1).alt
      i += 1
    }
    a
  }

  def nAltAlleles: Int = altAlleles.length

  def isBiallelic: Boolean = nAltAlleles == 1

  // FIXME altAllele, alt to be deprecated
  def altAllele: AltAllele = {
    require(isBiallelic, "called altAllele on a non-biallelic variant")
    altAlleles(0)
  }

  def alt: String = altAllele.alt

  def nAlleles: Int = 1 + nAltAlleles

  def allele(i: Int): String = if (i == 0)
    ref
  else
    altAlleles(i - 1).alt

  def nGenotypes = VariantMethods.nGenotypes(nAlleles)

  def locus: Locus = Locus(contig, start)

  def isAutosomalOrPseudoAutosomal(rg: RGBase): Boolean = isAutosomal(rg) || inXPar(rg) || inYPar(rg)

  def isAutosomal(rg: RGBase): Boolean = !(inX(rg) || inY(rg) || isMitochondrial(rg))

  def isMitochondrial(rg: RGBase): Boolean = rg.isMitochondrial(contig)

  def inXPar(rg: RGBase): Boolean = rg.inXPar(locus)

  def inYPar(rg: RGBase): Boolean = rg.inYPar(locus)

  def inXNonPar(rg: RGBase): Boolean = inX(rg) && !inXPar(rg)

  def inYNonPar(rg: RGBase): Boolean = inY(rg) && !inYPar(rg)

  private def inX(rg: RGBase): Boolean = rg.inX(contig)

  private def inY(rg: RGBase): Boolean = rg.inY(contig)

  import is.hail.variant.CopyState._

  def copyState(sex: Sex.Sex, rg: ReferenceGenome): CopyState =
    if (sex == Sex.Male)
      if (inXNonPar(rg))
        HemiX
      else if (inYNonPar(rg))
        HemiY
      else
        Auto
    else
      Auto

  def compare(that: Variant, rg: ReferenceGenome): Int = {
    val t = TStruct(
      "locus" -> TLocus(rg),
      "alleles" -> TArray(TString()))
    t.ordering.compare(Row(locus, alleles),
      Row(that.locus, that.alleles))
  }

  def minRep: Variant = {
    val (minLocus, minAlleles) = VariantMethods.minRep(locus, ref +: altAlleles.map(_.alt))
    Variant(minLocus.contig, minLocus.position, minAlleles(0), minAlleles.tail.toArray)
  }

  override def toString: String =
    s"$contig:$start:$ref:${ altAlleles.map(_.alt).mkString(",") }"

  def toRow = {
    Row.fromSeq(Array(
      contig,
      start,
      ref,
      altAlleles.map { a => Row.fromSeq(Array(a.ref, a.alt)) }))
  }

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("start", JInt(start)),
    ("ref", JString(ref)),
    ("altAlleles", JArray(altAlleles.map(_.toJSON).toList)))
}

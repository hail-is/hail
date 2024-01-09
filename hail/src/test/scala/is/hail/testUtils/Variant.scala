package is.hail.testUtils

import is.hail.annotations.Annotation
import is.hail.types.virtual.{TArray, TLocus, TString, TStruct}
import is.hail.variant._

import org.json4s._

import scala.collection.JavaConverters._

import org.apache.spark.sql.Row

object Variant {
  def apply(contig: String, start: Int, ref: String, alt: String): Variant =
    Variant(contig, start, ref, Array(AltAllele(ref, alt)))

  def apply(contig: String, start: Int, ref: String, alts: Array[String]): Variant =
    Variant(contig, start, ref, alts.map(alt => AltAllele(ref, alt)))

  def apply(contig: String, start: Int, ref: String, alts: Array[String], rg: ReferenceGenome)
    : Variant = {
    rg.checkLocus(contig, start)
    Variant(contig, start, ref, alts)
  }

  def apply(
    contig: String,
    start: Int,
    ref: String,
    alts: java.util.ArrayList[String],
    rg: ReferenceGenome,
  ): Variant = Variant(contig, start, ref, alts.asScala.toArray, rg)

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

case class Variant(contig: String, start: Int, ref: String, altAlleles: IndexedSeq[AltAllele]) {
  require(altAlleles.forall(_.ref == ref))

  /* The position is 1-based. Telomeres are indicated by using positions 0 or N+1, where N is the
   * length of the corresponding chromosome or contig. See the VCF spec, v4.2, section 1.4.1. */
  require(start >= 0, s"invalid variant: negative position: '${this.toString}'")
  require(!ref.isEmpty, s"invalid variant: empty contig: '${this.toString}'")

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

  def locus: Locus = Locus(contig, start)

  override def toString: String =
    s"$contig:$start:$ref:${altAlleles.map(_.alt).mkString(",")}"
}

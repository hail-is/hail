package is.hail.utils

import is.hail.annotations.{Annotation, RegionValue}
import is.hail.expr.{TStruct, TVariant, TArray, TString}
import is.hail.variant.{AltAllele, AltAlleleType, Variant, HardCallView}

import scala.collection.mutable

case class SummaryResult(samples: Int, variants: Long, callRate: Option[Double],
  contigs: Set[String], multiallelics: Long, snps: Long, mnps: Long, insertions: Long,
  deletions: Long, complex: Long, star: Long, maxAlleles: Int)

class SummaryCombiner(rowType: TStruct) extends Serializable {
  private val contigs = mutable.Set.empty[String]
  private var nCalled = 0L
  private var nVariants = 0L
  private var multiallelics = 0L
  private var maxAlleles = 2
  private var snps = 0L
  private var mnps = 0L
  private var insertions = 0L
  private var deletions = 0L
  private var complex = 0L
  private var star = 0L

  def merge(other: SummaryCombiner): SummaryCombiner = {

    contigs ++= other.contigs
    nCalled += other.nCalled
    nVariants += other.nVariants
    multiallelics += other.multiallelics
    maxAlleles = math.max(maxAlleles, other.maxAlleles)
    snps += other.snps
    mnps += other.mnps
    insertions += other.insertions
    deletions += other.deletions
    complex += other.complex
    star += other.star

    this
  }

  def merge(rv: RegionValue, view: HardCallView): SummaryCombiner = {
    val vType = rowType.fieldType(1).asInstanceOf[TVariant]
    val r = rv.region
    val vOffset = rowType.loadField(rv, 1)
    val t = vType.representation
    val altsType = t.fieldType(3).asInstanceOf[TArray]

    nVariants += 1

    val contig = TString.loadString(r, t.loadField(r, vOffset, 0))
    contigs += contig

    val altsOffset = t.loadField(r, vOffset, 3)
    val nAlts = altsType.loadLength(r, altsOffset)
    val nAlleles = nAlts + 1
    if (nAlleles > 2)
      multiallelics += 1
    if (nAlleles > maxAlleles)
      maxAlleles = nAlleles

    var i = 0
    while (i < nAlts) {
      val altOffset = altsType.loadElement(r, altsOffset, nAlts, i)
      val altAllele = AltAllele.fromRegionValue(r, altOffset)
      altAllele.altAlleleType match {
        case AltAlleleType.SNP => snps += 1
        case AltAlleleType.MNP => mnps += 1
        case AltAlleleType.Insertion => insertions += 1
        case AltAlleleType.Deletion => deletions += 1
        case AltAlleleType.Complex => complex += 1
        case AltAlleleType.Star => star += 1
      }
      i += 1
    }

    view.setRegion(rv)
    i = 0
    while (i < view.gsLength) {
      view.setGenotype(i)
      if (view.hasGT) nCalled += 1
      i += 1
    }

    this
  }

  def result(nSamples: Int): SummaryResult = SummaryResult(
    samples = nSamples,
    variants = nVariants,
    callRate = divOption(nCalled.toDouble, nSamples * nVariants),
    contigs = contigs.toSet,
    multiallelics = multiallelics,
    snps = snps,
    mnps = mnps,
    insertions = insertions,
    deletions = deletions,
    complex = complex,
    star = star,
    maxAlleles = maxAlleles)
}
package is.hail.utils

import is.hail.annotations.Annotation
import is.hail.variant.{AltAlleleType, GenomeReference, Variant}

import scala.collection.mutable

case class SummaryResult(samples: Int, variants: Long, callRate: Option[Double],
  contigs: Set[String], multiallelics: Long, snps: Long, mnps: Long, insertions: Long,
  deletions: Long, complex: Long, star: Long, maxAlleles: Int)

class SummaryCombiner[T](f: Iterable[T] => Int) extends Serializable {
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

  def merge(other: SummaryCombiner[T]): SummaryCombiner[T] = {

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

  def merge(data: (Variant, (Annotation, Iterable[T])), gr: GenomeReference): SummaryCombiner[T] = {
    nVariants += 1

    val v = data._1
    contigs += gr.contigNames(v.contig)
    if (v.nAlleles > 2)
      multiallelics += 1
    if (v.nAlleles > maxAlleles)
      maxAlleles = v.nAlleles

    v.altAlleles.foreach { aa =>
      aa.altAlleleType match {
        case AltAlleleType.SNP => snps += 1
        case AltAlleleType.MNP => mnps += 1
        case AltAlleleType.Insertion => insertions += 1
        case AltAlleleType.Deletion => deletions += 1
        case AltAlleleType.Complex => complex += 1
        case AltAlleleType.Star => star += 1
      }
    }

    nCalled += f(data._2._2)

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
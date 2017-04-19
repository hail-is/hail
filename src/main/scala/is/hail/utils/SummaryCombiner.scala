package is.hail.utils

import is.hail.annotations.Annotation
import is.hail.variant.Variant

import scala.collection.mutable

case class SummaryResult(samples: Int, variants: Long, callRate: Option[Double],
  contigs: Set[String], multiallelics: Long, snps: Long, indels: Long,
  complex: Long, mostAlleles: Int)

class SummaryCombiner[T](f: Iterable[T] => Int) extends Serializable {
  private val contigs = mutable.Set.empty[String]
  private var nCalled = 0L
  private var nVariants = 0L
  private var multiallelics = 0L
  private var mostAlleles = 2
  private var snps = 0L
  private var indels = 0L
  private var complex = 0L

  def merge(other: SummaryCombiner[T]): SummaryCombiner[T] = {

    contigs ++= other.contigs
    nCalled += other.nCalled
    nVariants += other.nVariants
    multiallelics += other.multiallelics
    mostAlleles = math.max(mostAlleles, other.mostAlleles)
    snps += other.snps
    indels += other.indels
    complex += other.complex

    this
  }

  def merge(data: (Variant, (Annotation, Iterable[T]))): SummaryCombiner[T] = {
    nVariants += 1

    val v = data._1
    contigs += v.contig
    if (v.nAlleles > 2)
      multiallelics += 1
    if (v.nAlleles > mostAlleles)
      mostAlleles = v.nAlleles

    v.altAlleles.foreach { aa =>
      if (aa.isSNP)
        snps += 1
      else if (aa.isIndel)
        indels += 1
      else
        complex += 1
    }

    nCalled += f(data._2._2)

    this
  }

  def result(nSamples: Int): SummaryResult = SummaryResult(
    nSamples,
    nVariants,
    divOption(nCalled.toDouble, nSamples * nVariants),
    contigs.toSet,
    multiallelics,
    snps,
    indels,
    complex,
    mostAlleles)
}
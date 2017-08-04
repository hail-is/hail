package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Genotype, Variant}

object CallStats {
  def schema = TStruct(Array(("AC", TArray(TInt), "Allele count. One element per allele **including reference**. There are two elements for a biallelic variant, or 4 for a variant with three alternate alleles."),
    ("AF", TArray(TDouble), "Allele frequency. One element per allele including reference. Sums to 1."),
    ("AN", TInt, "Allele number. This is equal to the sum of AC, or 2 * the total number of called genotypes in the aggregable."),
    ("GC", TArray(TInt), "Genotype count. One element per possible genotype, including reference genotypes -- 3 for biallelic, 6 for triallelic, 10 for 3 alt alleles, and so on. The sum of this array is the number of called genotypes in the aggregable.")
  ).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })
}

case class CallStats(alleleCount: IndexedSeq[Int], alleleFrequency: Option[IndexedSeq[Double]], alleleNumber: Int,
  genotypeCount: IndexedSeq[Int]) {
  require(alleleNumber == genotypeCount.sum * 2, s"AN was not equal to 2 * GC sum: $this")
  require(alleleFrequency.forall(f => D_==(f.sum, 1d)), s"AF did not sum to 1: $this")

  def asAnnotation: Annotation = Annotation(alleleCount, alleleFrequency.orNull, alleleNumber, genotypeCount)
}


class CallStatsCombiner(v: Variant) extends Serializable {
  val alleleCount = new Array[Int](v.nAlleles)
  val genotypeCount = new Array[Int](v.nGenotypes)

  def merge(g: Genotype): CallStatsCombiner = {
    Genotype.gt(g).foreach { gt =>
      val p = Genotype.gtPair(gt)
      alleleCount(p.j) += 1
      alleleCount(p.k) += 1
      genotypeCount(gt) += 1
    }
    this
  }

  def merge(that: CallStatsCombiner): CallStatsCombiner = {
    alleleCount.indices.foreach { i => alleleCount(i) += that.alleleCount(i) }
    genotypeCount.indices.foreach { i => genotypeCount(i) += that.genotypeCount(i) }
    this
  }

  def result(): CallStats = {
    val alleleNumber = alleleCount.sum
    val alleleFrequency =
      if (alleleNumber == 0)
        None
      else
        Some(alleleCount.map(_.toDouble / alleleNumber): IndexedSeq[Double])
    CallStats(alleleCount, alleleFrequency, alleleNumber, genotypeCount)
  }
}

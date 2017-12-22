package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.typ._
import is.hail.utils._
import is.hail.variant.{Call, Genotype, Variant}

object CallStats {
  def schema = TStruct(
    "AC" -> TArray(TInt32()),
    "AF" -> TArray(TFloat64()),
    "AN" -> TInt32(),
    "GC" -> TArray(TInt32()))
}

case class CallStats(alleleCount: IndexedSeq[Int], alleleFrequency: Option[IndexedSeq[Double]], alleleNumber: Int,
  genotypeCount: IndexedSeq[Int]) {
  require(alleleNumber == genotypeCount.sum * 2, s"AN was not equal to 2 * GC sum: $this")
  require(alleleFrequency.forall(f => D_==(f.sum, 1d)), s"AF did not sum to 1: $this")

  def asAnnotation: Annotation = Annotation(alleleCount, alleleFrequency.orNull, alleleNumber, genotypeCount)
}

class CallStatsCombiner(val v: Variant) extends Serializable {
  val alleleCount = new Array[Int](v.nAlleles)
  val genotypeCount = new Array[Int](v.nGenotypes)

  def merge(gt: Call): CallStatsCombiner = {
    if (gt != null) {
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
    assert(alleleCount.sum == 2 * genotypeCount.sum)

    val alleleNumber = alleleCount.sum
    val alleleFrequency =
      if (alleleNumber == 0)
        None
      else
        Some(alleleCount.map(_.toDouble / alleleNumber): IndexedSeq[Double])
    CallStats(alleleCount, alleleFrequency, alleleNumber, genotypeCount)
  }
}

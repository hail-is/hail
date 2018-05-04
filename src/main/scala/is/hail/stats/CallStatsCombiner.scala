package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.{Call, VariantMethods}

object CallStats {
  def schema = TStruct(
    "AC" -> TArray(TInt32()),
    "AF" -> TArray(TFloat64()),
    "AN" -> TInt32())
}

case class CallStats(alleleCount: IndexedSeq[Int], alleleFrequency: Option[IndexedSeq[Double]], alleleNumber: Int) {
  require(alleleFrequency.forall(f => D_==(f.sum, 1d)), s"AF did not sum to 1: $this")

  def asAnnotation: Annotation = Annotation(alleleCount, alleleFrequency.orNull, alleleNumber)
}

class CallStatsCombiner(val alleles: IndexedSeq[String]) extends Serializable {
  val alleleCount = new Array[Int](alleles.length)

  def merge(c: Call): CallStatsCombiner = {
    val p = Call.allelePair(c)
    alleleCount(p.j) += 1
    alleleCount(p.k) += 1
    this
  }

  def merge(that: CallStatsCombiner): CallStatsCombiner = {
    alleleCount.indices.foreach { i => alleleCount(i) += that.alleleCount(i) }
    this
  }

  def result(): CallStats = {
    val alleleNumber = alleleCount.sum
    val alleleFrequency =
      if (alleleNumber == 0)
        None
      else
        Some(alleleCount.map(_.toDouble / alleleNumber): IndexedSeq[Double])
    CallStats(alleleCount, alleleFrequency, alleleNumber)
  }
}

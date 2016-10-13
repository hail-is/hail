package org.broadinstitute.hail.stats

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, Variant}

object CallStats {
  def schema = TStruct(
    "AC" -> TArray(TInt),
    "AF" -> TArray(TDouble),
    "AN" -> TInt,
    "GC" -> TArray(TInt)
  )
}

case class CallStats(alleleCount: IndexedSeq[Int], alleleFrequency: IndexedSeq[Double], alleleNumber: Int,
  genotypeCount: IndexedSeq[Int]) {
  require(alleleNumber == genotypeCount.sum * 2, s"AN was not equal to 2 * GC sum: $this")
  require(D_==(alleleFrequency.sum, 1d), s"AF did not sum to 1: $this")


  def asAnnotation: Annotation = Annotation(alleleCount, alleleFrequency, alleleNumber, genotypeCount)
}


class CallStatsCombiner(v: Variant) extends Serializable {
  var alleleCount = new Array[Int](v.nAlleles)
  var genotypeCount = new Array[Int](v.nGenotypes)

  def alleleNumber = alleleCount.sum

  def alleleFrequency = alleleCount.map(ac => ac.toDouble / alleleNumber)

  def merge(g: Genotype): CallStatsCombiner = {
    g.gt.foreach { gt =>
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
    val alleleFrequency = alleleCount.map(_.toDouble / alleleNumber)
    CallStats(alleleCount, alleleFrequency, alleleNumber, genotypeCount)
  }
}

package is.hail.stats

import is.hail.annotations.{Annotation, RegionValueBuilder}
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.Call

import scala.annotation.switch

object CallStats {
  def schema = TStruct(
    "AC" -> TArray(TInt32()),
    "AF" -> TArray(TFloat64()),
    "AN" -> TInt32(),
    "homozygote_count" -> TArray(TInt32()))
}

case class CallStats(alleleCount: IndexedSeq[Int], alleleFrequency: Option[IndexedSeq[Double]], alleleNumber: Int,
  homCount: IndexedSeq[Int]) {
  require(alleleFrequency.forall(f => D_==(f.sum, 1d)), s"AF did not sum to 1: $this")
  require(homCount.sum >= 0 && homCount.sum <= alleleNumber.toDouble / 2)
  def asAnnotation: Annotation = Annotation(alleleCount, alleleFrequency.orNull, alleleNumber, homCount)
}

class CallStatsCombiner(val nAlleles: Int) extends Serializable {
  var alleleCount = new Array[Int](nAlleles)
  var homozygoteCount = new Array[Int](nAlleles)

  def merge(c: Call): CallStatsCombiner = {
    (Call.ploidy(c): @switch) match {
      case 0 =>
      case 1 =>
        alleleCount(Call.alleleByIndex(c, 0)) += 1
      case 2 =>
        val p = Call.allelePair(c)
        alleleCount(p.j) += 1
        alleleCount(p.k) += 1
        if (p.j == p.k)
          homozygoteCount(p.j) += 1
      case _ => throw new UnsupportedOperationException
    }
    this
  }

  def merge(that: CallStatsCombiner): CallStatsCombiner = {
    alleleCount.indices.foreach { i => alleleCount(i) += that.alleleCount(i) }
    homozygoteCount.indices.foreach { i => homozygoteCount(i) += that.homozygoteCount(i) }
    this
  }

  def result(): CallStats = {
    val alleleNumber = alleleCount.sum
    val alleleFrequency =
      if (alleleNumber == 0)
        None
      else
        Some(alleleCount.map(_.toDouble / alleleNumber): IndexedSeq[Double])
    CallStats(alleleCount, alleleFrequency, alleleNumber, homozygoteCount)
  }

  def result(rvb: RegionValueBuilder): Unit = {
    val cstats = result()
    rvb.startStruct()

    // AC
    rvb.startArray(cstats.alleleCount.length)
    cstats.alleleCount.foreach(rvb.addInt _)
    rvb.endArray()

    // AF
    cstats.alleleFrequency match {
      case Some(af) =>
        rvb.startArray(af.length)
        af.foreach(rvb.addDouble _)
        rvb.endArray()
      case None =>
        rvb.setMissing()
    }

    // AN
    rvb.addInt(cstats.alleleNumber)

    // homozygote_count
    rvb.startArray(cstats.homCount.length)
    cstats.homCount.foreach(rvb.addInt _)
    rvb.endArray()
    rvb.endStruct()
  }

  def copy(): CallStatsCombiner = {
    val c = new CallStatsCombiner(nAlleles)
    c.alleleCount = alleleCount.clone()
    c.homozygoteCount = homozygoteCount.clone()
    c
  }
}

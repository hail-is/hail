package org.broadinstitute.hail.variant

import org.apache.commons.math3.distribution.BinomialDistribution
import org.omg.PortableInterceptor.NON_EXISTENT
import org.scalacheck.{Gen, Arbitrary}

import scala.language.implicitConversions
import scala.collection.mutable
import org.broadinstitute.hail.Utils._

object GenotypeType extends Enumeration {
  type GenotypeType = Value
  val HomRef = Value(0)
  val Het = Value(1)
  val HomVar = Value(2)
  val NoCall = Value(-1)
}

import org.broadinstitute.hail.variant.GenotypeType.GenotypeType

case class Genotype(gt: Option[Int],
  ad: Option[IndexedSeq[Int]],
  dp: Option[Int],
  gq: Option[Int],
  pl: Option[IndexedSeq[Int]]) {

  require(gt.forall(v => v >= 0 && v <= 2))
  require(ad.forall(a => a.length == 2))
  require(pl.forall(a => a.length == 3))

  def write(b: mutable.ArrayBuilder[Byte]) {
    b += ((gt.getOrElse(-1) & 7)
      | (if (ad.isDefined) 0x8 else 0)
      | (if (dp.isDefined) 0x10 else 0)
      | (if (gq.isDefined) 0x20 else 0)
      | (if (pl.isDefined) 0x40 else 0)).toByte

    ad.foreach(_.foreach(b.writeULEB128))
    dp.foreach(b.writeULEB128)
    gq.foreach(b.writeULEB128)
    pl.foreach(_.foreach(b.writeULEB128))
  }

  def isHomRef: Boolean = gt.contains(0)

  def isHet: Boolean = gt.contains(1)

  def isHomVar: Boolean = gt.contains(2)

  def isNonRef: Boolean = gt.exists(_ >= 1)

  def isNotCalled: Boolean = gt.isEmpty

  def isCalled: Boolean = gt.isDefined

  def gtType: GenotypeType = GenotypeType(gt.getOrElse(-1))


  def nNonRef: Int = gt.getOrElse(0)

  override def toString: String = {
    val b = new StringBuilder

    gt match {
      case Some(0) => b.append("0/0")
      case Some(1) => b.append("0/1")
      case Some(2) => b.append("1/1")
      case None => b.append("./.")
    }

    b += ':'
    b.append(ad.map(_.mkString(",")).getOrElse("."))
    b += ':'
    b.append(dp.map(_.toString).getOrElse("."))
    b += ':'
    b.append(gq.map(_.toString).getOrElse("."))
    b += ':'
    b.append(pl.map(_.mkString(",")).getOrElse("."))

    b.result()
  }

  def pAB(theta: Double = 0.5): Option[Double] = ad.map { case IndexedSeq(refDepth, altDepth) =>
    val d = new BinomialDistribution(refDepth + altDepth, theta)
    val minDepth = refDepth.min(altDepth)
    val minp = d.probability(minDepth)
    val mincp = d.cumulativeProbability(minDepth)
    (2 * mincp - minp).min(1.0).max(0.0)
  }

  def gtString(v: Variant): String = {
    if (isHomRef)
      v.ref + "/" + v.ref
    else if (isHet)
      v.ref + "/" + v.alt
    else if (isHomVar)
      v.alt + "/" + v.alt
    else
      "./."
  }
}

object Genotype {
  def read(a: Iterator[Byte]): Genotype = {
    val flags = a.next()

    val gtv = (flags << 29) >> 29
    val gt = if (gtv == -1)
      None
    else
      Some(gtv)

    val ad: Option[IndexedSeq[Int]] = if ((flags & 0x8) != 0)
      Some(Array[Int](a.readULEB128(),
        a.readULEB128()))
    else
      None

    val dp = if ((flags & 0x10) != 0)
      Some(a.readULEB128())
    else
      None

    val gq = if ((flags & 0x20) != 0)
      Some(a.readULEB128())
    else
      None

    val pl: Option[IndexedSeq[Int]] = if ((flags & 0x40) != 0)
      Some(Array[Int](a.readULEB128(),
        a.readULEB128(),
        a.readULEB128()))
    else
      None

    Genotype(gt, ad, dp, gq, pl)
  }

  implicit def arbGenotype: Arbitrary[Genotype] = Arbitrary {
    for {gt <- Gen.choose(-1, 2)
      ad1 <- Gen.choose(0, 1000)
      ad2 <- Gen.choose(0, 1000)
      dpDelta <- Gen.choose(0, 100)
      gq <- Gen.choose(0, 99)
      pl1 <- Gen.choose(0, 10000)
      pl2 <- Gen.choose(0, 10000)}
      yield Genotype(if (gt == -1)
        None
      else
        Some(gt),
        Some(Array(ad1, ad2)),
        Some(ad1 + ad2 + dpDelta),
        Some(gq),
        gt match {
          case -1 => None
          case 0 => Some(Array(0, pl1, pl2))
          case 1 => Some(Array(pl1, 0, pl2))
          case 2 => Some(Array(pl1, pl2, 0))
        })
  }
}
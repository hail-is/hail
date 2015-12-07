package org.broadinstitute.hail.variant

import org.apache.commons.math3.distribution.BinomialDistribution
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

case class Genotype(gt: Option[Int] = None,
  ad: Option[IndexedSeq[Int]] = None,
  dp: Option[Int] = None,
  pl: Option[IndexedSeq[Int]] = None) {
  require(gt.forall(_ >= 0))

  def check(v: Variant) {
    check(v.nAlleles)
  }

  def check(nAlleles: Int) {
    val nGenotypes = Variant.nGenotypes(nAlleles)
    assert(gt.forall(i => i >= 0 && i < nGenotypes))
    assert(ad.forall(a => a.length == nAlleles))
    assert(pl.forall(a => a.length == nGenotypes))
  }

  def isHomRef: Boolean = gt.contains(0)

  // FIXME i, j => j, k
  def isHet: Boolean = gt.exists { gt =>
    val (i, j) = Genotype.gtPair(gt)
    i != j
  }

  def isHomVar: Boolean = gt.exists(gt =>
    gt > 0 && {
      val (i, j) = Genotype.gtPair(gt)
      i == j
    })

  def isNonRef: Boolean = gt.exists(_ > 0)

  def isHetNonRef: Boolean = gt.exists { gt =>
    val (i, j) = Genotype.gtPair(gt)
    i > 0 && i != j
  }

  def isHetRef: Boolean = gt.exists { gt =>
    val (i, j) = Genotype.gtPair(gt)
    i == 0 && j > 0
  }

  def isNotCalled: Boolean = gt.isEmpty

  def isCalled: Boolean = gt.isDefined

  def gtType: GenotypeType = GenotypeType(gt.getOrElse(-1))

  def nNonRef: Int = gt.getOrElse(0)

  def gq: Option[Int] = gt.flatMap(gtx =>
    pl.map { plx =>
      if (plx.length < 2)
      // FIXME
        Int.MaxValue
      else
        plx.indices
          .filter(_ != gtx)
          .map(i => plx(i))
          .min
    })

  override def toString: String = {
    val b = new StringBuilder

    b.append(gt.map { gt =>
      val (i, j) = Genotype.gtPair(gt)
      s"$i/$j"
    }.getOrElse("."))
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
}

object Genotype {
  def apply(gtx: Int): Genotype = Genotype(Some(gtx))

  final val flagMultiHasGT = 0x1
  final val flagMultiGTRef = 0x2
  final val flagBiGTMask = 0x3
  final val flagHasAD = 0x4
  final val flagHasDP = 0x8
  final val flagHasPL = 0x10
  final val flagADSimple = 0x20
  final val flagDPSimple = 0x40
  // 0x80 reserved

  // FIXME speed up, cache small, sqrt
  def gtPair(t: Int): (Int, Int) = {
    def f(i: Int, j: Int): (Int, Int) = if (i <= j)
      (i, j)
    else
      f(i - j - 1, j + 1)

    f(t, 0)
  }

  def gtIndex(i: Int, j: Int): Int = {
    require(i >= 0 && i <= j)
    j * (j + 1) / 2 + i
  }

  def read(v: Variant, a: Iterator[Byte]): Genotype =
    read(v.nAlleles, a)

  def read(nAlleles: Int, a: Iterator[Byte]): Genotype = {
    val isBiallelic = nAlleles == 2
    val nGenotypes = Variant.nGenotypes(nAlleles)

    val flags = a.next()

    val gt: Option[Int] =
      if (isBiallelic) {
        if ((flags & Genotype.flagBiGTMask) == 0)
          None
        else
          Some((flags & Genotype.flagBiGTMask) - 1)
      } else {
        if ((flags & Genotype.flagMultiHasGT) != 0) {
          if ((flags & Genotype.flagMultiGTRef) != 0)
            Some(0)
          else
            Some(a.readULEB128())
        } else
          None
      }

    val ad: Option[IndexedSeq[Int]] =
      if ((flags & Genotype.flagHasAD) != 0) {
        if ((flags & Genotype.flagADSimple) != 0) {
          assert(gt.isDefined)
          val (j, k) = Genotype.gtPair(gt.get)
          val adx = Array.fill(nAlleles)(0)
          adx(j) = a.readULEB128()
          if (j != k)
            adx(k) = a.readULEB128()
          Some(adx)
        } else
          Some(Array.fill(nAlleles)(a.readULEB128()))
      } else
        None

    val dp =
      if ((flags & Genotype.flagHasDP) != 0) {
        if (ad.isDefined) {
          if ((flags & Genotype.flagDPSimple) != 0)
            Some(ad.get.sum)
          else
            Some(ad.get.sum + a.readSLEB128())
        } else
          Some(a.readULEB128())
      } else
        None

    val pl: Option[IndexedSeq[Int]] =
      if ((flags & Genotype.flagHasPL) != 0) {
        if (gt.isDefined) {
          val pla = Array.fill(nGenotypes)(0)
          val gtx = gt.get
          for (i <- 0 until gtx)
            pla(i) = a.readULEB128()
          for (i <- (gtx + 1) until pla.length)
            pla(i) = a.readSLEB128()
          Some(pla)
        } else
          Some(Array.fill(nGenotypes)(a.readULEB128()))
      } else
        None

    Genotype(gt, ad, dp, pl)
  }

  def gen(nAlleles: Int): Gen[Genotype] = {
    val nGenotypes = Variant.nGenotypes(nAlleles)

    for (gt: Option[Int] <- genOption(Gen.choose(0, nGenotypes - 1));
      ad <- genOption(Gen.buildableOfN[IndexedSeq[Int], Int](nAlleles, genNonnegInt));
      dp <- genOption(Gen.posNum[Int]);
      pl: Option[Array[Int]] <- genOption(Gen.buildableOfN[Array[Int], Int](nGenotypes, genNonnegInt))) yield {
      gt.foreach { gtx =>
        pl.foreach { pla => pla(gtx) = 0 }
      }
      val g = Genotype(gt, ad, dp, pl.map(pla => pla: IndexedSeq[Int]))
      g.check(nAlleles)
      g
    }
  }

  def gen(v: Variant): Gen[Genotype] = gen(v.nAlleles)

  def genWithSize: Gen[(Int, Genotype)] =
    for (nAlleles <- Gen.choose(1, 10);
      g <- gen(nAlleles))
      yield (nAlleles, g)

  def gen: Gen[Genotype] =
    for (nAlleles <- Gen.choose(1, 10);
      g <- gen(nAlleles))
      yield g

  implicit def arbGenotype = Arbitrary(gen)
}

class GenotypeBuilder(nAlleles: Int) {
  def this(v: Variant) = this(v.nAlleles)

  val isBiallelic = nAlleles == 2
  val nGenotypes = Variant.nGenotypes(nAlleles)

  private var flags: Int = 0

  private var gt: Int = 0
  private var ad: IndexedSeq[Int] = _
  private var dp: Int = 0
  private var pl: IndexedSeq[Int] = _

  def clear() {
    flags = 0
  }

  def setGT(newGT: Int) {
    require(newGT >= 0 && newGT <= nGenotypes)
    if (isBiallelic) {
      assert((flags & Genotype.flagBiGTMask) == 0)
      flags = flags | (newGT + 1)
    } else {
      assert((flags & Genotype.flagMultiHasGT) == 0)
      flags |= Genotype.flagMultiHasGT
      if (newGT == 0)
        flags |= Genotype.flagMultiGTRef
    }

    gt = newGT
  }

  def setAD(newAD: IndexedSeq[Int]) {
    require(newAD.length == nAlleles)
    flags |= Genotype.flagHasAD
    ad = newAD
  }

  def setDP(newDP: Int) {
    flags |= Genotype.flagHasDP
    dp = newDP
  }

  def setPL(newPL: IndexedSeq[Int]) {
    require(newPL.length == nGenotypes)
    flags |= Genotype.flagHasPL
    pl = newPL
  }

  def set(g: Genotype) {
    g.gt.foreach(setGT)
    g.ad.foreach(setAD)
    g.dp.foreach(setDP)
    g.pl.foreach(setPL)
  }

  def write(b: mutable.ArrayBuilder[Byte]) {
    val hasGT = if (isBiallelic)
      (flags & Genotype.flagBiGTMask) != 0
    else
      (flags & Genotype.flagMultiHasGT) != 0

    val hasAD = (flags & Genotype.flagHasAD) != 0
    val hasDP = (flags & Genotype.flagHasDP) != 0
    val hasPL = (flags & Genotype.flagHasPL) != 0

    var j = 0
    var k = 0
    var adsum = 0

    if (hasGT) {
      // FIXME
      val p = Genotype.gtPair(gt)
      j = p._1
      k = p._2
      if (hasAD) {
        if (ad.indices
          .filter(i => i != j && i != k)
          .forall(i => ad(i) == 0))
          flags |= Genotype.flagADSimple
      }
    }

    if (hasAD && hasDP) {
      adsum = ad.sum
      if (adsum == dp)
        flags |= Genotype.flagDPSimple
    }

    b += flags.toByte
    if (hasGT && !isBiallelic && gt != 0)
      b.writeULEB128(gt)

    if (hasAD) {
      if ((flags & Genotype.flagADSimple) != 0) {
        b.writeULEB128(ad(j))
        if (k != j)
          b.writeULEB128(ad(k))
      } else
        ad.foreach(b.writeULEB128)
    }

    if (hasDP) {
      if (hasAD) {
        if ((flags & Genotype.flagDPSimple) == 0)
          b.writeSLEB128(dp - adsum)
      } else
        b.writeULEB128(dp)
    }

    if (hasPL) {
      if (hasGT) {
        assert(pl(gt) == 0)
        for (i <- 0 until gt)
          b.writeULEB128(pl(i))
        for (i <- gt + 1 until pl.length)
          b.writeSLEB128(pl(i))
      } else
        pl.foreach(b.writeSLEB128)
    }
  }

}

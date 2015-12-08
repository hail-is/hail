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

object GTPair {
  def apply(j: Int, k: Int): GTPair = {
    require(j >= 0 && j <= 0xffff)
    require(k >= 0 && k <= 0xffff)
    new GTPair(j | (k << 16))
  }
}

class GTPair(val p: Int) extends AnyVal {
  def j: Int = p & 0xffff

  def k: Int = (p >> 16) & 0xffff
}

class Genotype(private val flags: Byte,
  private val _gt: Int,
  private val _ad: Array[Int],
  private val _dp: Int,
  private val _pl: Array[Int]) extends Serializable {

  require(_gt >= -1)
  require((_gt == -1) == ((flags & Genotype.flagHasGT) == 0))

  override def equals(that: Any): Boolean = that match {
    case g: Genotype =>
      flags == g.flags &&
        _gt == g._gt &&
        (_ad == null || _ad.sameElements(g._ad)) &&
        _dp == g._dp &&
        (_pl == null || _pl.sameElements(g._pl))
    case _ => false
  }

  override def hashCode: Int = {
    flags ^ _gt ^ _dp ^
      (if (_ad != null) _ad.hashCode else 0) ^
      (if (_pl != null) _pl.hashCode else 0)
  }

  def gt: Option[Int] =
    if (_gt >= 0)
      Some(_gt)
    else
      None

  def ad: Option[IndexedSeq[Int]] =
    if ((flags & Genotype.flagHasAD) != 0)
      Some(_ad)
    else
      None

  def dp: Option[Int] =
    if ((flags & Genotype.flagHasDP) != 0)
      Some(_dp)
    else
      None

  def pl: Option[IndexedSeq[Int]] =
    if ((flags & Genotype.flagHasPL) != 0)
      Some(_pl)
    else
      None

  def check(v: Variant) {
    check(v.nAlleles)
  }

  def check(nAlleles: Int) {
    val nGenotypes = Variant.nGenotypes(nAlleles)
    assert(gt.forall(i => i >= 0 && i < nGenotypes))
    assert(ad.forall(a => a.length == nAlleles))
    assert(pl.forall(a => a.length == nGenotypes))
  }

  def isHomRef: Boolean = _gt == 0

  // FIXME i, j => j, k
  def isHet: Boolean = _gt >= 0 && {
    val p = Genotype.gtPair(_gt)
    p.j != p.k
  }

  def isHomVar: Boolean = _gt > 0 && {
    val p = Genotype.gtPair(_gt)
    p.j == p.k
  }

  def isNonRef: Boolean = _gt >= 0

  def isHetNonRef: Boolean = _gt >= 0 && {
    val p = Genotype.gtPair(_gt)
    p.j > 0 && p.j != p.k
  }

  def isHetRef: Boolean = _gt >= 0 && {
    val p = Genotype.gtPair(_gt)
    p.j == 0 && p.k > 0
  }

  def isNotCalled: Boolean = _gt == -1

  def isCalled: Boolean = _gt >= 0

  // FIXME NO
  def gtType: GenotypeType = GenotypeType(gt.getOrElse(-1))

  def nNonRef: Int =
    if (_gt >= 0) {
      val p = Genotype.gtPair(_gt)
      if (p.j == p.k)
        1
      else
        2
    } else
      0

  def gq: Option[Int] =
    if (_gt >= 0 && ((flags & Genotype.flagHasPL) != 0)) {
      var r = Int.MaxValue
      for (i <- 0 until _gt)
        r = r.min(_pl(i))
      for (i <- (_gt + 1) until _pl.length)
        r = r.min(_pl(i))
      Some(r)
    } else
      None

  override def toString: String = {
    val b = new StringBuilder

    b.append(gt.map { gt =>
      val p = Genotype.gtPair(gt)
      s"${p.j}/${p.k}"
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
  def apply(gtx: Int): Genotype = new Genotype(flagHasGT.toByte, gtx, null, 0, null)

  def apply(gt: Option[Int] = None,
    ad: Option[IndexedSeq[Int]] = None,
    dp: Option[Int] = None,
    pl: Option[IndexedSeq[Int]] = None): Genotype = {

    val flags =
      ((if (gt.isDefined) Genotype.flagHasGT else 0)
        | (if (ad.isDefined) Genotype.flagHasAD else 0)
        | (if (dp.isDefined) Genotype.flagHasDP else 0)
        | (if (pl.isDefined) Genotype.flagHasPL else 0))

    new Genotype(flags.toByte, gt.getOrElse(-1), ad.map(_.toArray).orNull, dp.getOrElse(0), pl.map(_.toArray).orNull)
  }

  final val flagMultiHasGT = 0x1
  final val flagMultiGTRef = 0x2
  final val flagBiGTMask = 0x3

  // only used by expanded genotype
  final val flagHasGT = 0x1

  final val flagHasAD = 0x4
  final val flagHasDP = 0x8
  final val flagHasPL = 0x10
  final val flagADSimple = 0x20
  final val flagDPSimple = 0x40
  // 0x80 reserved

  val smallGTPair = Array(new GTPair(0x0), new GTPair(0x10000), new GTPair(0x10001),
    new GTPair(0x20000), new GTPair(0x20001), new GTPair(0x20002),
    new GTPair(0x30000), new GTPair(0x30001), new GTPair(0x30002), new GTPair(0x30003),
    new GTPair(0x40000), new GTPair(0x40001), new GTPair(0x40002), new GTPair(0x40003), new GTPair(0x40004),
    new GTPair(0x50000), new GTPair(0x50001), new GTPair(0x50002), new GTPair(0x50003), new GTPair(0x50004),
    new GTPair(0x50005),
    new GTPair(0x60000), new GTPair(0x60001), new GTPair(0x60002), new GTPair(0x60003), new GTPair(0x60004),
    new GTPair(0x60005), new GTPair(0x60006),
    new GTPair(0x70000), new GTPair(0x70001), new GTPair(0x70002), new GTPair(0x70003), new GTPair(0x70004),
    new GTPair(0x70005), new GTPair(0x70006), new GTPair(0x70007))

  // FIXME speed up, cache small, sqrt
  def gtPair(i: Int): GTPair = {
    def f(j: Int, k: Int): GTPair = if (j <= k)
      GTPair(j, k)
    else
      f(j - k - 1, k + 1)

    if (i <= 35)
      smallGTPair(i)
    else
      f(i, 0)
  }

  def gtIndex(j: Int, k: Int): Int = {
    require(j >= 0 && j <= k)
    k * (k + 1) / 2 + j
  }

  def gtIndex(p: GTPair): Int = gtIndex(p.j, p.k)

  def read(v: Variant, a: Iterator[Byte]): Genotype =
    read(v.nAlleles, a)

  def read(nAlleles: Int, a: Iterator[Byte]): Genotype = {
    val isBiallelic = nAlleles == 2
    val nGenotypes = Variant.nGenotypes(nAlleles)

    val flags: Byte = a.next()
    var newFlags: Int = flags & (Genotype.flagHasAD | Genotype.flagHasDP | Genotype.flagHasPL)

    val gt: Int =
      if (isBiallelic) {
        if ((flags & Genotype.flagBiGTMask) == 0)
          -1 // None
        else {
          newFlags |= Genotype.flagHasGT
          (flags & Genotype.flagBiGTMask) - 1
        }
      } else {
        if ((flags & Genotype.flagMultiHasGT) != 0) {
          newFlags |= Genotype.flagHasGT
          if ((flags & Genotype.flagMultiGTRef) != 0)
            0
          else
            a.readULEB128()
        } else
          -1 // None
      }

    val ad: Array[Int] =
      if ((flags & Genotype.flagHasAD) != 0) {
        val ada = new Array[Int](nAlleles)
        if ((flags & Genotype.flagADSimple) != 0) {
          assert((newFlags & Genotype.flagHasGT) != 0)
          val p = Genotype.gtPair(gt)
          ada(p.j) = a.readULEB128()
          if (p.j != p.k)
            ada(p.k) = a.readULEB128()
        } else {
          for (i <- ada.indices)
            ada(i) = a.readULEB128()
        }
        ada
      } else
        null

    val dp =
      if ((flags & Genotype.flagHasDP) != 0) {
        if ((newFlags & Genotype.flagHasAD) != 0) {
          if ((flags & Genotype.flagDPSimple) != 0)
            ad.sum
          else
            ad.sum + a.readSLEB128()
        } else
          a.readULEB128()
      } else
        0 // None

    val pl: Array[Int] =
      if ((flags & Genotype.flagHasPL) != 0) {
        val pla = new Array[Int](nGenotypes)
        if ((newFlags & Genotype.flagHasGT) != 0) {
          var i = 0
          while (i < gt) {
            pla(i) = a.readULEB128()
            i += 1
          }
          i += 1
          while (i < pla.length) {
            pla(i) = a.readULEB128()
            i += 1
          }
          pla
        } else {
          var i = 0
          while (i < pla.length) {
            pla(i) = a.readULEB128()
            i += 1
          }
        }
        pla
      } else
        null

    new Genotype(newFlags.toByte, gt, ad, dp, pl)
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
      j = p.j
      k = p.k
      if (hasAD) {
        if (ad.indices
          .filter(i => i != p.j && i != p.k)
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
          b.writeULEB128(pl(i))
      } else
        pl.foreach(b.writeULEB128)
    }
  }

}

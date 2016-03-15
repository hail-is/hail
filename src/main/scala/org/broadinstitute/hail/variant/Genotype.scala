package org.broadinstitute.hail.variant

import java.util
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.apache.commons.math3.distribution.BinomialDistribution
import org.apache.spark.sql.types._
import org.broadinstitute.hail.ByteIterator
import org.broadinstitute.hail.check.{Gen, Arbitrary}
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

  def nNonRefAlleles: Int =
    (if (j != 0) 1 else 0) + (if (k != 0) 1 else 0)
}

object Foo {
  def f[T](i: Iterable[T], j: Iterable[T]) = i.zip(j)
}

class Genotype(private val _gt: Int,
  private val _ad: Array[Int],
  private val _dp: Int,
  private val _gq: Int,
  private val _pl: Array[Int],
  val fakeRef: Boolean) extends Serializable {

  require(_gt >= -1)
  require(_dp >= -1)

  def check(v: Variant) {
    assert(gt.forall(i => i >= 0 && i < v.nGenotypes))
    assert(ad.forall(a => a.length == v.nAlleles))
    assert(pl.forall(a => a.length == v.nGenotypes))
  }

  def copy(gt: Option[Int] = this.gt,
    ad: Option[Array[Int]] = this.ad,
    dp: Option[Int] = this.dp,
    gq: Option[Int] = this.gq,
    pl: Option[Array[Int]] = this.pl,
    fakeRef: Boolean = this.fakeRef): Genotype = Genotype(gt, ad, dp, gq, pl, fakeRef)

  override def equals(that: Any): Boolean = that match {
    case g: Genotype =>
      _gt == g._gt &&
        ((_ad == null && g._ad == null)
          || (_ad != null && g._ad != null && _ad.sameElements(g._ad))) &&
        _dp == g._dp &&
        _gq == g._gq &&
        ((_pl == null && g._pl == null)
          || (_pl != null && g._pl != null && _pl.sameElements(g._pl)))

    case _ => false
  }

  override def hashCode: Int =
    new HashCodeBuilder(43, 19)
      .append(_gt)
      .append(util.Arrays.hashCode(_ad))
      .append(_dp)
      .append(_gq)
      .append(util.Arrays.hashCode(_pl))
      .append(fakeRef)
      .toHashCode

  def gt: Option[Int] =
    if (_gt >= 0)
      Some(_gt)
    else
      None

  def ad: Option[Array[Int]] = Option(_ad)

  def dp: Option[Int] =
    if (_dp >= 0)
      Some(_dp)
    else
      None

  def od: Option[Int] =
    if (_dp >= 0 && _ad != null)
      Some(_dp - _ad.sum)
    else
      None

  def gq: Option[Int] =
    if (_gq >= 0)
      Some(_gq)
    else
      None

  def pl: Option[Array[Int]] = Option(_pl)

  def isHomRef: Boolean = _gt == 0

  def isHet: Boolean = _gt > 0 && {
    val p = Genotype.gtPair(_gt)
    p.j != p.k
  }

  def isHomVar: Boolean = _gt > 0 && {
    val p = Genotype.gtPair(_gt)
    p.j == p.k
  }

  def isCalledNonRef: Boolean = _gt > 0

  def isHetNonRef: Boolean = _gt > 0 && {
    val p = Genotype.gtPair(_gt)
    p.j > 0 && p.j != p.k
  }

  def isHetRef: Boolean = _gt > 0 && {
    val p = Genotype.gtPair(_gt)
    p.j == 0 && p.k > 0
  }

  def isNotCalled: Boolean = _gt == -1

  def isCalled: Boolean = _gt >= 0

  def gtType: GenotypeType =
    if (isHomRef)
      GenotypeType.HomRef
    else if (isHet)
      GenotypeType.Het
    else if (isHomVar)
      GenotypeType.HomVar
    else {
      assert(isNotCalled)
      GenotypeType.NoCall
    }

  def nNonRefAlleles: Option[Int] =
    if (_gt >= 0)
      Some(Genotype.gtPair(_gt).nNonRefAlleles)
    else
      None

  override def toString: String = {
    val b = new StringBuilder

    b.append(gt.map { gt =>
      val p = Genotype.gtPair(gt)
      s"${p.j}/${p.k}"
    }.getOrElse("./."))
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

  def pAB(theta: Double = 0.5): Option[Double] = ad.map { case Array(refDepth, altDepth) =>
    val d = new BinomialDistribution(refDepth + altDepth, theta)
    val minDepth = refDepth.min(altDepth)
    val minp = d.probability(minDepth)
    val mincp = d.cumulativeProbability(minDepth)
    (2 * mincp - minp).min(1.0).max(0.0)
  }
}

object Genotype {
  def apply(gtx: Int): Genotype = new Genotype(gtx, null, -1, -1, null, false)

  def apply(gt: Option[Int] = None,
    ad: Option[Array[Int]] = None,
    dp: Option[Int] = None,
    gq: Option[Int] = None,
    pl: Option[Array[Int]] = None,
    fakeRef: Boolean = false): Genotype = {
    new Genotype(gt.getOrElse(-1), ad.map(_.toArray).orNull, dp.getOrElse(-1), gq.getOrElse(-1), pl.map(_.toArray).orNull, fakeRef)
  }

  def schema: DataType = StructType(Array(
    StructField("gt", IntegerType),
    StructField("ad", ArrayType(IntegerType)),
    StructField("dp", IntegerType),
    StructField("gq", IntegerType),
    StructField("pl", ArrayType(IntegerType)),
    StructField("fakeRef", BooleanType)))

  final val flagMultiHasGTBit = 0x1
  final val flagMultiGTRefBit = 0x2
  final val flagBiGTMask = 0x3
  final val flagHasADBit = 0x4
  final val flagHasDPBit = 0x8
  final val flagHasGQBit = 0x10
  final val flagHasPLBit = 0x20
  final val flagSimpleADBit = 0x40
  final val flagSimpleDPBit = 0x80
  final val flagSimpleGQBit = 0x100
  final val flagFakeRefBit = 0x200

  def flagHasGT(isBiallelic: Boolean, flags: Int) =
    if (isBiallelic)
      (flags & flagBiGTMask) != 0
    else
      (flags & flagMultiHasGTBit) != 0

  def flagStoresGT(isBiallelic: Boolean, flags: Int) =
    isBiallelic || ((flags & flagMultiGTRefBit) != 0)

  def flagGT(isBiallelic: Boolean, flags: Int) = {
    assert(flagStoresGT(isBiallelic, flags))
    if (isBiallelic)
      (flags & flagBiGTMask) - 1
    else {
      assert((flags & flagMultiGTRefBit) != 0)
      0
    }
  }

  def flagSetGT(isBiallelic: Boolean, flags: Int, gt: Int): Int = {
    if (isBiallelic) {
      assert(gt >= 0 && gt <= 2)
      flags | ((gt & flagBiGTMask) + 1)
    } else {
      if (gt == 0)
        flags | flagMultiHasGTBit | flagMultiGTRefBit
      else
        flags | flagMultiHasGTBit
    }
  }

  def flagHasAD(flags: Int): Boolean = (flags & flagHasADBit) != 0

  def flagHasDP(flags: Int): Boolean = (flags & flagHasDPBit) != 0

  def flagHasGQ(flags: Int): Boolean = (flags & flagHasGQBit) != 0

  def flagHasPL(flags: Int): Boolean = (flags & flagHasPLBit) != 0

  def flagSetHasAD(flags: Int): Int = flags | flagHasADBit

  def flagSetHasDP(flags: Int): Int = flags | flagHasDPBit

  def flagSetHasGQ(flags: Int): Int = flags | flagHasGQBit

  def flagSetHasPL(flags: Int): Int = flags | flagHasPLBit

  def flagSimpleAD(flags: Int): Boolean = (flags & flagSimpleADBit) != 0

  def flagSimpleDP(flags: Int): Boolean = (flags & flagSimpleDPBit) != 0

  def flagSimpleGQ(flags: Int): Boolean = (flags & flagSimpleGQBit) != 0

  def flagSetSimpleAD(flags: Int): Int = flags | flagSimpleADBit

  def flagSetSimpleDP(flags: Int): Int = flags | flagSimpleDPBit

  def flagSetSimpleGQ(flags: Int): Int = flags | flagSimpleGQBit

  def flagFakeRef(flags: Int): Boolean = (flags & flagFakeRefBit) != 0

  def flagSetFakeRef(flags: Int): Int = flags | flagFakeRefBit

  def gqFromPL(pl: Array[Int]): Int = {
    var m = 99
    var m2 = 99
    var i = 0
    while (i < pl.length) {
      if (pl(i) < m) {
        m2 = m
        m = pl(i)
      } else if (pl(i) < m2)
        m2 = pl(i)
      i += 1
    }
    assert(m == 0)
    m2
  }

  val smallGTPair = Array(GTPair(0, 0), GTPair(0, 1), GTPair(1, 1),
    GTPair(0, 2), GTPair(1, 2), GTPair(2, 2),
    GTPair(0, 3), GTPair(1, 3), GTPair(2, 3), GTPair(3, 3),
    GTPair(0, 4), GTPair(1, 4), GTPair(2, 4), GTPair(3, 4), GTPair(4, 4),
    GTPair(0, 5), GTPair(1, 5), GTPair(2, 5), GTPair(3, 5), GTPair(4, 5), GTPair(5, 5),
    GTPair(0, 6), GTPair(1, 6), GTPair(2, 6), GTPair(3, 6), GTPair(4, 6), GTPair(5, 6),
    GTPair(6, 6),
    GTPair(0, 7), GTPair(1, 7), GTPair(7, 2), GTPair(3, 7), GTPair(4, 7),
    GTPair(5, 7), GTPair(7, 6), GTPair(7, 7))

  def gtPairRecursive(i: Int): GTPair = {
    def f(j: Int, k: Int): GTPair = if (j <= k)
      GTPair(j, k)
    else
      f(j - k - 1, k + 1)

    f(i, 0)
  }

  def gtPairSqrt(i: Int): GTPair = {
    val k: Int = (Math.sqrt(8 * i.toDouble + 1) / 2 - 0.5).toInt
    assert(k * (k + 1) / 2 <= i)
    val j = i - k * (k + 1) / 2
    assert(gtIndex(j, k) == i)
    GTPair(j, k)
  }

  def gtPair(i: Int): GTPair = {
    if (i < smallGTPair.length)
      smallGTPair(i)
    else
      gtPairSqrt(i)
  }

  def gtIndex(j: Int, k: Int): Int = {
    require(j >= 0 && j <= k)
    k * (k + 1) / 2 + j
  }

  def gtIndex(p: GTPair): Int = gtIndex(p.j, p.k)

  def read(v: Variant, a: ByteIterator): Genotype = {
    val isBiallelic = v.isBiallelic

    val flags = a.readULEB128()

    val gt: Int =
      if (flagHasGT(isBiallelic, flags)) {
        if (flagStoresGT(isBiallelic, flags))
          flagGT(isBiallelic, flags)
        else
          a.readULEB128()
      } else
        -1

    val ad: Array[Int] =
      if (flagHasAD(flags)) {
        val ada = new Array[Int](v.nAlleles)
        if (flagSimpleAD(flags)) {
          assert(gt >= 0)
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
      if (flagHasDP(flags)) {
        if (flagHasAD(flags)) {
          if (flagSimpleDP(flags))
            ad.sum
          else
            ad.sum + a.readULEB128()
        } else
          a.readULEB128()
      } else
        -1 // None

    val pl: Array[Int] =
      if (flagHasPL(flags)) {
        val pla = new Array[Int](v.nGenotypes)
        if (gt >= 0) {
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

    val gq: Int =
      if (flagHasGQ(flags)) {
        if (flagSimpleGQ(flags))
          gqFromPL(pl)
        else
          a.readULEB128()
      } else
        -1

    new Genotype(gt, ad, dp, gq, pl, flagFakeRef(flags))
  }

  def gen(v: Variant): Gen[Genotype] = {
    val m = Int.MaxValue / (v.nAlleles + 1)
    for (gt: Option[Int] <- Gen.option(Gen.choose(0, v.nGenotypes - 1));
      ad <- Gen.option(Gen.buildableOfN[Array[Int], Int](v.nAlleles,
        Gen.choose(0, m)));
      dp <- Gen.option(Gen.choose(0, m));
      gq <- Gen.option(Gen.choose(0, 10000));
      pl <- Gen.option(Gen.buildableOfN[Array[Int], Int](v.nGenotypes,
        Gen.choose(0, m)))) yield {
      gt.foreach { gtx =>
        pl.foreach { pla => pla(gtx) = 0 }
      }
      pl.foreach { pla =>
        val m = pla.min
        var i =  0
        while (i < pla.length) {
          pla(i) -= m
          i += 1
        }
      }
      val g = Genotype(gt, ad,
        dp.map(_ + ad.map(_.sum).getOrElse(0)), gq, pl)
      g.check(v)
      g
    }
  }

  def genVariantGenotype: Gen[(Variant, Genotype)] =
    for (v <- Variant.gen;
      g <- gen(v))
      yield (v, g)

  def genArb: Gen[Genotype] =
    for (v <- Variant.gen;
      g <- gen(v))
      yield g

  implicit def arbGenotype = Arbitrary(genArb)
}

class GenotypeBuilder(v: Variant) {

  val isBiallelic = v.isBiallelic
  val nGenotypes = v.nGenotypes

  var flags: Int = 0

  private var gt: Int = 0
  private var ad: Array[Int] = _
  private var dp: Int = 0
  private var gq: Int = 0
  private var pl: Array[Int] = _

  def clear() {
    flags = 0
  }

  def hasGT: Boolean =
    Genotype.flagHasGT(isBiallelic, flags)

  def setGT(newGT: Int) {
    require(newGT >= 0 && newGT <= nGenotypes)
    require(!hasGT)
    flags = Genotype.flagSetGT(isBiallelic, flags, newGT)
    gt = newGT
  }

  def setAD(newAD: Array[Int]) {
    require(newAD.length == v.nAlleles)
    flags = Genotype.flagSetHasAD(flags)
    ad = newAD
  }

  def setDP(newDP: Int) {
    assert(newDP >= 0)
    flags = Genotype.flagSetHasDP(flags)
    dp = newDP
  }

  def setGQ(newGQ: Int) {
    assert(newGQ >= 0)
    flags = Genotype.flagSetHasGQ(flags)
    gq = newGQ
  }

  def setPL(newPL: Array[Int]) {
    require(newPL.length == v.nGenotypes)
    flags = Genotype.flagSetHasPL(flags)
    pl = newPL
  }

  def setFakeRef() {
    flags = Genotype.flagSetFakeRef(flags)
  }

  def set(g: Genotype) {
    g.gt.foreach(setGT)
    g.ad.foreach(setAD)
    g.dp.foreach(setDP)
    g.gq.foreach(setGQ)
    g.pl.foreach(setPL)
    if (g.fakeRef)
      setFakeRef()
  }

  def write(b: mutable.ArrayBuilder[Byte]) {
    val hasGT = Genotype.flagHasGT(isBiallelic, flags)

    val hasAD = Genotype.flagHasAD(flags)
    val hasDP = Genotype.flagHasDP(flags)
    val hasGQ = Genotype.flagHasGQ(flags)
    val hasPL = Genotype.flagHasPL(flags)

    var j = 0
    var k = 0
    if (hasGT) {
      val p = Genotype.gtPair(gt)
      j = p.j
      k = p.k
      if (hasAD) {
        var i = 0
        var simple = true
        while (i < ad.length && simple) {
          if (i != j && i != k && ad(i) != 0)
            simple = false
          i += 1
        }
        if (simple)
          flags = Genotype.flagSetSimpleAD(flags)
      }
    }

    var adsum = 0
    if (hasAD && hasDP) {
      adsum = ad.sum
      if (adsum == dp)
        flags = Genotype.flagSetSimpleDP(flags)
    }

    if (hasPL && hasGQ) {
      val gqFromPL = Genotype.gqFromPL(pl)
      if (gq == gqFromPL)
        flags = Genotype.flagSetSimpleGQ(flags)
    }

    /*
    println("flags:")
    if (Genotype.flagHasGT(isBiallelic, flags))
      println(s"  gt = $gt")
    if (Genotype.flagHasDP(flags))
      println(s"  dp = $dp")
    */

    b.writeULEB128(flags)

    if (hasGT && !Genotype.flagStoresGT(isBiallelic, flags))
      b.writeULEB128(gt)

    if (hasAD) {
      if (Genotype.flagSimpleAD(flags)) {
        b.writeULEB128(ad(j))
        if (k != j)
          b.writeULEB128(ad(k))
      } else
        ad.foreach(b.writeULEB128)
    }

    if (hasDP) {
      if (hasAD) {
        if (!Genotype.flagSimpleDP(flags))
          b.writeULEB128(dp - adsum)
      } else
        b.writeULEB128(dp)
    }

    if (hasPL) {
      if (hasGT) {
        var i = 0
        while (i < gt) {
          b.writeULEB128(pl(i))
          i += 1
        }
        i += 1
        while (i < pl.length) {
          b.writeULEB128(pl(i))
          i += 1
        }
      } else
        pl.foreach(b.writeULEB128)
    }

    if (hasGQ) {
      if (!Genotype.flagSimpleGQ(flags))
        b.writeULEB128(gq)
    }
  }
}

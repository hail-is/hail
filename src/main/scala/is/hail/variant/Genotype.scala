package is.hail.variant

import java.util

import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.{TArray, TBoolean, TInt, TStruct, Type}
import is.hail.utils.{ByteIterator, _}
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.apache.commons.math3.distribution.BinomialDistribution
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

import scala.language.implicitConversions
import scala.reflect.ClassTag

object GenotypeType extends Enumeration {
  type GenotypeType = Value
  val HomRef = Value(0)
  val Het = Value(1)
  val HomVar = Value(2)
  val NoCall = Value(-1)
}

import is.hail.variant.GenotypeType.GenotypeType

object GTPair {
  def apply(j: Int, k: Int): GTPair = {
    require(j >= 0 && j <= 0xffff, s"GTPair invalid j value $j")
    require(k >= 0 && k <= 0xffff, s"GTPair invalid k value $k")
    new GTPair(j | (k << 16))
  }

  def fromNonNormalized(j: Int, k: Int): GTPair = {
    if (j <= k)
      GTPair(j, k)
    else
      GTPair(k, j)
  }
}

class GTPair(val p: Int) extends AnyVal {
  def j: Int = p & 0xffff

  def k: Int = (p >> 16) & 0xffff

  def nNonRefAlleles: Int =
    (if (j != 0) 1 else 0) + (if (k != 0) 1 else 0)

  def alleleIndices: Array[Int] = {
    Array(this.j, this.k)
  }

}


abstract class Genotype extends Serializable {

  def unboxedGT: Int

  def unboxedAD: Array[Int]

  def unboxedDP: Int

  def unboxedGQ: Int

  def unboxedPX: Array[Int]

  def fakeRef: Boolean

  def isLinearScale: Boolean

  def unboxedPL: Array[Int] =
    if (unboxedPX == null)
      null
    else if (isLinearScale)
      Genotype.linearToPhred(unboxedPX)
    else
      unboxedPX

  def unboxedGP: Array[Double] =
    if (unboxedPX == null)
      null
    else if (isLinearScale)
      unboxedPX.map(_ * Genotype.gpNorm)
    else
      Genotype.plToGP(unboxedPX)

  def unboxedDosage: Double =
    if (unboxedPX == null)
      -1d
    else if (unboxedPX.size != 3)
      fatal("Genotype dosage is not defined for multi-allelic variants")
    else if (isLinearScale)
      (unboxedPX(1) + 2 * unboxedPX(2)) * Genotype.gpNorm
    else
      Genotype.plToDosage(unboxedPX(0), unboxedPX(1), unboxedPX(2))

  def check(nAlleles: Int) {
    val nGenotypes = triangle(nAlleles)
    assert(gt.forall(i => i >= 0 && i < nGenotypes))
    assert(ad.forall(a => a.length == nAlleles))
    assert(px.forall(a => a.length == nGenotypes))
  }

  def copy(gt: Option[Int] = this.gt,
    ad: Option[Array[Int]] = this.ad,
    dp: Option[Int] = this.dp,
    gq: Option[Int] = this.gq,
    px: Option[Array[Int]] = this.px,
    fakeRef: Boolean = this.fakeRef,
    isLinearScale: Boolean = this.isLinearScale): Genotype = Genotype(gt, ad, dp, gq, px, fakeRef, isLinearScale)

  override def equals(that: Any): Boolean = that match {
    case g: Genotype =>
      unboxedGT == g.unboxedGT &&
        ((ad.isEmpty && g.ad.isEmpty) || (ad.isDefined && g.ad.isDefined && ad.get.sameElements(g.ad.get))) &&
        dp == g.dp &&
        gq == g.gq &&
        ((px.isEmpty && g.px.isEmpty) || (px.isDefined && g.px.isDefined && px.get.sameElements(g.px.get))) &&
        fakeRef == g.fakeRef &&
        isLinearScale == g.isLinearScale

    case _ => false
  }

  override def hashCode: Int =
    new HashCodeBuilder(43, 19)
      .append(unboxedGT)
      .append(util.Arrays.hashCode(ad.orNull))
      .append(dp)
      .append(gq)
      .append(util.Arrays.hashCode(px.orNull))
      .append(fakeRef)
      .append(isLinearScale)
      .toHashCode

  def call: Call =
    if (unboxedGT == -1)
      null
    else
      Call(box(unboxedGT))

  def gt: Option[Int] =
    if (unboxedGT == -1)
      None
    else
      Some(unboxedGT)

  def ad: Option[Array[Int]] = Option(unboxedAD)

  def dp: Option[Int] =
    if (unboxedDP == -1)
      None
    else
      Some(unboxedDP)

  def hasOD: Boolean = unboxedDP != -1 && unboxedAD != null

  def od_ : Int = unboxedDP - intArraySum(unboxedAD)

  def od: Option[Int] =
    if (hasOD)
      Some(od_)
    else
      None

  def gq: Option[Int] =
    if (unboxedGQ == -1)
      None
    else
      Some(unboxedGQ)

  def px: Option[Array[Int]] = Option(unboxedPX)

  def pl: Option[Array[Int]] = Option(unboxedPL)

  def gp: Option[Array[Double]] = Option(unboxedGP)

  def dosage: Option[Double] =
    if (unboxedDosage == -1)
      None
    else
      Some(unboxedDosage)

  def isHomRef: Boolean = unboxedGT == 0

  def isHet: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j != p.k
  }

  def isHomVar: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j == p.k
  }

  def isCalledNonRef: Boolean = unboxedGT > 0

  def isHetNonRef: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j > 0 && p.j != p.k
  }

  def isHetRef: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j == 0 && p.k > 0
  }

  def isNotCalled: Boolean = unboxedGT == -1

  def isCalled: Boolean = unboxedGT >= 0

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

  def hasNNonRefAlleles: Boolean = unboxedGT != -1

  def nNonRefAlleles_ : Int = Genotype.gtPair(unboxedGT).nNonRefAlleles

  def nNonRefAlleles: Option[Int] =
    if (hasNNonRefAlleles)
      Some(nNonRefAlleles_)
    else
      None

  def fractionReadsRef(): Option[Double] =
    if (unboxedAD != null) {
      val s = intArraySum(unboxedAD)
      if (s != 0)
        Some(unboxedAD(0).toDouble / s)
      else
        None
    } else
      None

  def oneHotAlleles(nAlleles: Int): Option[IndexedSeq[Int]] = {
    gt.map { call =>
      val gtPair = Genotype.gtPair(call)
      val j = gtPair.j
      val k = gtPair.k
      new IndexedSeq[Int] with Serializable {
        def length: Int = nAlleles

        def apply(idx: Int): Int = {
          if (idx < 0 || idx >= nAlleles)
            throw new ArrayIndexOutOfBoundsException(idx)
          var r = 0
          if (idx == j)
            r += 1
          if (idx == k)
            r += 1
          r
        }
      }
    }
  }

  def oneHotAlleles(v: Variant): Option[IndexedSeq[Int]] = oneHotAlleles(v.nAlleles)

  def oneHotGenotype(v: Variant): Option[IndexedSeq[Int]] = oneHotGenotype(v.nGenotypes)

  def oneHotGenotype(nGenotypes: Int): Option[IndexedSeq[Int]] = {
    gt.map { call =>
      new IndexedSeq[Int] with Serializable {
        def length: Int = nGenotypes

        def apply(idx: Int): Int = {
          if (idx < 0 || idx >= nGenotypes)
            throw new ArrayIndexOutOfBoundsException(idx)
          if (idx == call)
            1
          else
            0
        }
      }
    }
  }

  override def toString: String = {
    val b = new StringBuilder

    b.append(gt.map { gt =>
      val p = Genotype.gtPair(gt)
      s"${ p.j }/${ p.k }"
    }.getOrElse("./."))

    if (fakeRef) {
      b += '*'
    }

    b += ':'
    b.append(ad.map(_.mkString(",")).getOrElse("."))
    b += ':'
    b.append(dp.map(_.toString).getOrElse("."))
    b += ':'
    b.append(gq.map(_.toString).getOrElse("."))
    b += ':'
    if (!isLinearScale) {
      b.append("PL=" + pl.map(_.mkString(",")).getOrElse("."))
    } else {
      b.append("GP=" + gp.map(_.mkString(",")).getOrElse("."))
    }

    b.result()
  }

  def hasPAB: Boolean = unboxedAD != null && isHet

  def pAB_(theta: Double = 0.5): Double = {
    val gtPair = Genotype.gtPair(unboxedGT)
    val aDepth = unboxedAD(gtPair.j)
    val bDepth = unboxedAD(gtPair.k)
    val d = new BinomialDistribution(aDepth + bDepth, theta)
    val minDepth = aDepth.min(bDepth)
    val minp = d.probability(minDepth)
    val mincp = d.cumulativeProbability(minDepth)
    (2 * mincp - minp).min(1.0).max(0.0)
  }

  def pAB(theta: Double = 0.5): Option[Double] =
    if (hasPAB)
      Some(pAB_(theta))
    else
      None

  def toRow: Row = Row(
    if (unboxedGT == -1) null else unboxedGT,
    if (unboxedAD == null) null else unboxedAD: IndexedSeq[Int],
    if (unboxedDP == -1) null else unboxedDP,
    if (unboxedGQ == -1) null else unboxedGQ,
    if (unboxedPX == null) null else unboxedPX: IndexedSeq[Int],
    fakeRef,
    isLinearScale)

  def toJSON: JValue = JObject(
    ("gt", gt.map(JInt(_)).getOrElse(JNull)),
    ("ad", ad.map(ads => JArray(ads.map(JInt(_)).toList)).getOrElse(JNull)),
    ("dp", dp.map(JInt(_)).getOrElse(JNull)),
    ("gq", gq.map(JInt(_)).getOrElse(JNull)),
    ("px", px.map(pxs => JArray(pxs.map(JInt(_)).toList)).getOrElse(JNull)),
    ("fakeRef", JBool(fakeRef)),
    ("isLinearScale", JBool(isLinearScale)))
}

class RowGenotype(r: Row) extends Genotype {
  def unboxedGT: Int =
    if (r.isNullAt(0))
      -1
    else
      r.getInt(0)

  def unboxedAD: Array[Int] =
    if (r.isNullAt(1))
      null
    else
      r.getSeq(1).toArray

  def unboxedDP: Int =
    if (r.isNullAt(2))
      -1
    else
      r.getInt(2)

  def unboxedGQ: Int =
    if (r.isNullAt(3))
      -1
    else
      r.getInt(3)

  def unboxedPX: Array[Int] =
    if (r.isNullAt(4))
      null
    else
      r.getSeq(4).toArray

  // not nullable
  def fakeRef: Boolean = r.getBoolean(5)

  def isLinearScale: Boolean = r.getBoolean(6)
}

object Genotype {
  def apply(gtx: Int): Genotype = new GenericGenotype(gtx, null, -1, -1, null, false, false)

  def apply(call: Call): Genotype = {
    val gtx: Int = if (call == null) -1 else call
    Genotype(gtx)
  }

  def apply(nAlleles: Int, gt: java.lang.Integer, dos: Array[Double]): Genotype = {
    val gtx: Int = if (gt == null) -1 else gt
    val g = new DosageGenotype(gtx, if (dos == null) null else Genotype.weightsToLinear(dos))
    g.check(nAlleles)
    g
  }

  def apply(nAlleles: Int, dos: Array[Double]): Genotype = {
    val px = if (dos == null) null else Genotype.weightsToLinear(dos)
    val gtx = if (dos == null) -1 else Genotype.unboxedGTFromLinear(px)
    val g = new DosageGenotype(gtx, px)
    g.check(nAlleles)
    g
  }

  def apply(nAlleles: Int, gt: java.lang.Integer, ad: Array[Int], dp: java.lang.Integer, gq: java.lang.Integer, pl: Array[Int]): Genotype = {
    val gtx: Int = if (gt == null) -1 else gt
    val dpx: Int = if (dp == null) -1 else dp
    val gqx: Int = if (gq == null) -1 else gq

    val g = new GenericGenotype(gtx, ad, dpx, gqx, pl, fakeRef = false, isLinearScale = false)
    g.check(nAlleles)
    g
  }

  def apply(gt: Option[Int], fakeRef: Boolean): Genotype =
    new GenericGenotype(gt.getOrElse(-1), null, -1, -1, null, fakeRef, false)

  def apply(gt: Option[Int] = None,
    ad: Option[Array[Int]] = None,
    dp: Option[Int] = None,
    gq: Option[Int] = None,
    px: Option[Array[Int]] = None,
    fakeRef: Boolean = false,
    isLinearScale: Boolean = false): Genotype = {
    new GenericGenotype(gt.getOrElse(-1), ad.map(_.toArray).orNull, dp.getOrElse(-1), gq.getOrElse(-1), px.map(_.toArray).orNull, fakeRef, isLinearScale)
  }

  def sparkSchema: DataType = StructType(Array(
    StructField("gt", IntegerType),
    StructField("ad", ArrayType(IntegerType, containsNull = false)),
    StructField("dp", IntegerType),
    StructField("gq", IntegerType),
    StructField("px", ArrayType(IntegerType, containsNull = false)),
    StructField("fakeRef", BooleanType, nullable = false),
    StructField("isLinearScale", BooleanType, nullable = false)))

  def expandedType: TStruct = TStruct(
    "gt" -> TInt,
    "ad" -> TArray(TInt),
    "dp" -> TInt,
    "gq" -> TInt,
    "px" -> TArray(TInt),
    "fakeRef" -> TBoolean,
    "isLinearScale" -> TBoolean)

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

  def flagHasGT(isBiallelic: Boolean, flags: Int): Boolean =
    if (isBiallelic)
      (flags & flagBiGTMask) != 0
    else
      (flags & flagMultiHasGTBit) != 0

  def flagStoresGT(isBiallelic: Boolean, flags: Int): Boolean =
    isBiallelic || ((flags & flagMultiGTRefBit) != 0)

  def flagGT(isBiallelic: Boolean, flags: Int): Int = {
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

  def flagHasPX(flags: Int): Boolean = (flags & flagHasPLBit) != 0

  def flagSetHasAD(flags: Int): Int = flags | flagHasADBit

  def flagSetHasDP(flags: Int): Int = flags | flagHasDPBit

  def flagSetHasGQ(flags: Int): Int = flags | flagHasGQBit

  def flagSetHasPX(flags: Int): Int = flags | flagHasPLBit

  def flagSimpleAD(flags: Int): Boolean = (flags & flagSimpleADBit) != 0

  def flagSimpleDP(flags: Int): Boolean = (flags & flagSimpleDPBit) != 0

  def flagSimpleGQ(flags: Int): Boolean = (flags & flagSimpleGQBit) != 0

  def flagSetSimpleAD(flags: Int): Int = flags | flagSimpleADBit

  def flagSetSimpleDP(flags: Int): Int = flags | flagSimpleDPBit

  def flagSetSimpleGQ(flags: Int): Int = flags | flagSimpleGQBit

  def flagFakeRef(flags: Int): Boolean = (flags & flagFakeRefBit) != 0

  def flagSetFakeRef(flags: Int): Int = flags | flagFakeRefBit

  def flagUnsetFakeRef(flags: Int): Int = flags ^ flagFakeRefBit

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
    assert(m == 0, s"$m, $m2, [${ pl.mkString(",") }]")
    m2
  }

  def gtFromLinear(a: Array[Int]): Option[Int] = {
    def f(i: Int, m: Int, mi: Int, count: Int): Option[Int] = {
      if (i == a.length) {
        assert(count >= 1)
        if (count == 1)
          Some(mi)
        else
          None
      } else if (a(i) > m)
        f(i + 1, a(i), i, 1)
      else if (a(i) == m)
        f(i + 1, m, mi, count + 1)
      else
        f(i + 1, m, mi, count)
    }

    f(1, a(0), 0, 1)
  }

  def unboxedGTFromLinear(a: Array[Int]): Int = {
    def f(i: Int, m: Int, mi: Int, count: Int): Int = {
      if (i == a.length) {
        assert(count >= 1)
        if (count == 1)
          mi
        else
          -1
      } else if (a(i) > m)
        f(i + 1, a(i), i, 1)
      else if (a(i) == m)
        f(i + 1, m, mi, count + 1)
      else
        f(i + 1, m, mi, count)
    }

    f(1, a(0), 0, 1)
  }

  def weightsToLinear(a: ArrayUInt): Array[Int] = {
    val n = a.length
    val r = new Array[Int](a.length)
    val s = a.sum.toDouble
    assert(s >= 0)
    var aAcc = 0.0
    var rAcc = 0
    var i = 0
    while (i < n) {
      aAcc += a(i).toDouble
      val t = (aAcc * 32768 / s + 0.5).toInt
      r(i) = t - rAcc
      rAcc = t
      i += 1
    }
    assert(rAcc == 32768)
    r
  }

  def weightsToLinear[T: Numeric](a: Array[T]): Array[Int] = {
    import scala.math.Numeric.Implicits._

    val n = a.length
    val r = new Array[Int](a.length)
    val s = a.sum.toDouble
    assert(s >= 0)
    var aAcc = 0.0
    var rAcc = 0
    var i = 0
    while (i < n) {
      aAcc += a(i).toDouble
      val t = (aAcc * 32768 / s + 0.5).toInt
      r(i) = t - rAcc
      rAcc = t
      i += 1
    }
    assert(rAcc == 32768)
    r
  }

  def weightsToLinear(w0: Int, w1: Int, w2: Int): Array[Int] = {
    val sum = w0 + w1 + w2
    assert(sum > 0)

    val l0 = (w0.toDouble * 32768 / sum + 0.5).toInt
    val l1 = ((w0 + w1).toDouble * 32768 / sum + 0.5).toInt - l0
    val l2 = 32768 - l0 - l1
    Array(l0, l1, l2)
  }

  def weightsToLinear(w0: Double, w1: Double, w2: Double): Array[Int] = {
    val sum = w0 + w1 + w2
    assert(sum > 0)

    val l0 = (w0 * 32768 / sum + 0.5).toInt
    val l1 = ((w0 + w1) * 32768 / sum + 0.5).toInt - l0
    val l2 = 32768 - l0 - l1
    Array(l0, l1, l2)
  }

  val gpNorm: Double = 1 / 32768.0

  lazy val linearToPhredConversionTable: Array[Double] = (0 to 65535).map { i => -10 * math.log10(if (i == 0) .25 else i) }.toArray

  def linearToPhred(a: Array[Int]): Array[Int] = {
    val x = a.map(linearToPhredConversionTable)
    x.map { d => (d - x.min + 0.5).toInt }
  }

  val maxPhredInTable = 8192

  lazy val phredToLinearConversionTable: Array[Double] = (0 to maxPhredInTable).map { i => math.pow(10, i / -10.0) }.toArray

  def phredToLinear(i: Int): Double =
    if (i < maxPhredInTable) phredToLinearConversionTable(i) else math.pow(10, i / -10.0)

  def plToGP(a: Array[Int]): Array[Double] = {
    val lkhd = a.map(i => phredToLinear(i))
    val s = lkhd.sum
    lkhd.map(_ / s)
  }

  def plToDosage(px0: Int, px1: Int, px2: Int): Double = {
    val p0 = phredToLinear(px0)
    val p1 = phredToLinear(px1)
    val p2 = phredToLinear(px2)

    (p1 + 2 * p2) / (p0 + p1 + p2)
  }

  val smallGTPair = Array(GTPair(0, 0), GTPair(0, 1), GTPair(1, 1),
    GTPair(0, 2), GTPair(1, 2), GTPair(2, 2),
    GTPair(0, 3), GTPair(1, 3), GTPair(2, 3), GTPair(3, 3),
    GTPair(0, 4), GTPair(1, 4), GTPair(2, 4), GTPair(3, 4), GTPair(4, 4),
    GTPair(0, 5), GTPair(1, 5), GTPair(2, 5), GTPair(3, 5), GTPair(4, 5), GTPair(5, 5),
    GTPair(0, 6), GTPair(1, 6), GTPair(2, 6), GTPair(3, 6), GTPair(4, 6), GTPair(5, 6),
    GTPair(6, 6),
    GTPair(0, 7), GTPair(1, 7), GTPair(2, 7), GTPair(3, 7), GTPair(4, 7),
    GTPair(5, 7), GTPair(6, 7), GTPair(7, 7))

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
    require(j >= 0 && j <= k, s"invalid gtIndex: ($j, $k)")
    k * (k + 1) / 2 + j
  }

  def gtIndex(p: GTPair): Int = gtIndex(p.j, p.k)

  def gtIndexWithSwap(i: Int, j: Int): Int = {
    if (j < i)
      gtIndex(j, i)
    else
      gtIndex(i, j)
  }

  def read(nAlleles: Int, isLinearScale: Boolean, a: ByteIterator): Genotype = {
    val isBiallelic = nAlleles == 2

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
        val ada = new Array[Int](nAlleles)
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
          var i = 0
          var adsum = 0
          while (i < ad.length) {
            adsum += ad(i)
            i += 1
          }
          if (flagSimpleDP(flags))
            adsum
          else
            adsum + a.readULEB128()
        } else
          a.readULEB128()
      } else
        -1 // None

    val px: Array[Int] =
      if (flagHasPX(flags)) {
        val pxa = new Array[Int](triangle(nAlleles))
        if (gt >= 0) {
          var i = 0
          while (i < gt) {
            pxa(i) = a.readULEB128()
            i += 1
          }
          i += 1
          while (i < pxa.length) {
            pxa(i) = a.readULEB128()
            i += 1
          }

          if (isLinearScale)
            pxa(gt) = 32768 - pxa.sum // original values summed to 32768 or 1.0 in probability

        } else {
          var i = 0
          while (i < pxa.length) {
            pxa(i) = a.readULEB128()
            i += 1
          }
        }

        pxa
      } else
        null

    val gq: Int =
      if (flagHasGQ(flags)) {
        if (flagSimpleGQ(flags))
          gqFromPL(px)
        else
          a.readULEB128()
      } else
        -1

    new GenericGenotype(gt, ad, dp, gq, px, flagFakeRef(flags), isLinearScale)
  }

  def readHardCall(nAlleles: Int, isLinearScale: Boolean, a: ByteIterator): Int = {
    val isBiallelic = nAlleles == 2

    val flags = a.readULEB128()

    val gt: Int =
      if (flagHasGT(isBiallelic, flags)) {
        if (flagStoresGT(isBiallelic, flags))
          flagGT(isBiallelic, flags)
        else
          a.readULEB128()
      } else
        -1

    var count = 0

    if (flagHasAD(flags)) {
      if (flagSimpleAD(flags)) {
        count += 1
        val p = Genotype.gtPair(gt)
        if (p.j != p.k)
          count += 1
      } else
        count += nAlleles
    }

    if (flagHasDP(flags) && !(flagHasAD(flags) && flagSimpleDP(flags)))
      count += 1

    if (flagHasPX(flags))
      if (gt >= 0)
        count += triangle(nAlleles) - 1
      else
        count += triangle(nAlleles)

    if (flagHasGQ(flags) && !flagSimpleGQ(flags))
      count += 1

    a.skipLEB128(count)

    gt
  }

  def readDosage(isLinearScale: Boolean, a: ByteIterator): Double = {
    val nAlleles = 2
    val isBiallelic = true

    val flags = a.readULEB128()

    val gt: Int =
      if (flagHasGT(isBiallelic, flags)) {
        if (flagStoresGT(isBiallelic, flags))
          flagGT(isBiallelic, flags)
        else
          a.readULEB128()
      } else
        -1

    var count = 0

    if (flagHasAD(flags)) {
      if (flagSimpleAD(flags)) {
        count += 1
        val p = Genotype.gtPair(gt)
        if (p.j != p.k)
          count += 1
      } else
        count += nAlleles
    }

    if (flagHasDP(flags) && !(flagHasAD(flags) && flagSimpleDP(flags)))
      count += 1

    a.skipLEB128(count)

    val dosage: Double =
      if (flagHasPX(flags)) {
        var px0 = 0
        var px1 = 0
        var px2 = 0

        if (gt == 0) {
          px1 = a.readULEB128()
          px2 = a.readULEB128()
          if (isLinearScale) px0 = 32768 - (px1 + px2)
        } else if (gt == 1) {
          px0 = a.readULEB128()
          px2 = a.readULEB128()
          if (isLinearScale) px1 = 32768 - (px0 + px2)
        } else if (gt == 2) {
          px0 = a.readULEB128()
          px1 = a.readULEB128()
          if (isLinearScale) px2 = 32768 - (px0 + px1)
        } else {
          px0 = a.readULEB128()
          px1 = a.readULEB128()
          px2 = a.readULEB128()
        }

        if (isLinearScale)
          (px1 + 2 * px2) * Genotype.gpNorm
        else
          Genotype.plToDosage(px0, px1, px2)
      } else
        -1d

    if (flagHasGQ(flags) && !flagSimpleGQ(flags))
      a.skipLEB128(1)

    dosage
  }

  def genDosageGenotype(v: Variant): Gen[Genotype] = {
    val nAlleles = v.nAlleles
    val nGenotypes = triangle(nAlleles)
    for (px <- Gen.option(Gen.partition(nGenotypes, 32768))) yield {
      val gt = px.flatMap(gtFromLinear)
      val g = Genotype(gt = gt, px = px, isLinearScale = true)
      g.check(nAlleles)
      g
    }
  }

  def genExtreme(v: Variant): Gen[Genotype] = {
    val nAlleles = v.nAlleles
    val m = Int.MaxValue / (nAlleles + 1)
    val nGenotypes = triangle(nAlleles)
    for (gt: Option[Int] <- Gen.option(Gen.choose(0, nGenotypes - 1));
      ad <- Gen.option(Gen.buildableOfN[Array, Int](nAlleles, Gen.choose(0, m)));
      dp <- Gen.option(Gen.choose(0, m));
      gq <- Gen.option(Gen.choose(0, 10000));
      pl <- Gen.oneOfGen(
        Gen.option(Gen.buildableOfN[Array, Int](nGenotypes, Gen.choose(0, m))),
        Gen.option(Gen.buildableOfN[Array, Int](nGenotypes, Gen.choose(0, 100))))) yield {
      gt.foreach { gtx =>
        pl.foreach { pla => pla(gtx) = 0 }
      }
      pl.foreach { pla =>
        val m = pla.min
        var i = 0
        while (i < pla.length) {
          pla(i) -= m
          i += 1
        }
      }
      val g = Genotype(gt, ad, dp.map(_ + ad.map(_.sum).getOrElse(0)), gq, pl)
      g.check(nAlleles)
      g
    }
  }

  def genRealistic(v: Variant): Gen[Genotype] = {
    val nAlleles = v.nAlleles
    val nGenotypes = triangle(nAlleles)
    for (callRate <- Gen.choose(0d, 1d);
      alleleFrequencies <- Gen.buildableOfN[Array, Double](nAlleles, Gen.choose(1e-6, 1d)) // avoid divison by 0
        .map { rawWeights =>
        val sum = rawWeights.sum
        rawWeights.map(_ / sum)
      };
      gt <- Gen.option(Gen.zip(Gen.chooseWithWeights(alleleFrequencies), Gen.chooseWithWeights(alleleFrequencies))
        .map { case (gti, gtj) => gtIndexWithSwap(gti, gtj) }, callRate);
      ad <- Gen.option(Gen.buildableOfN[Array, Int](nAlleles,
        Gen.choose(0, 50)));
      dp <- Gen.choose(0, 30).map(d => ad.map(o => o.sum + d));
      pl <- Gen.option(Gen.buildableOfN[Array, Int](nGenotypes, Gen.choose(0, 1000)).map { arr =>
        gt match {
          case Some(i) =>
            arr(i) = 0
            arr
          case None =>
            val min = arr.min
            arr.map(_ - min)
        }
      });
      gq <- Gen.choose(-30, 30).map(i => pl.map(pls => math.max(0, gqFromPL(pls) + i)))
    ) yield
      Genotype(gt, ad, dp, gq, pl)
  }

  def genVariantGenotype: Gen[(Variant, Genotype)] =
    for (v <- Variant.gen;
      g <- Gen.oneOfGen(genExtreme(v), genRealistic(v), genDosageGenotype(v)))
      yield (v, g)

  def genArb: Gen[Genotype] =
    for (v <- Variant.gen;
      g <- Gen.oneOfGen(genExtreme(v), genRealistic(v), genDosageGenotype(v)))
      yield g

  implicit def arbGenotype = Arbitrary(genArb)

  implicit val ordering: Ordering[Genotype] =
    new Ordering[Genotype] {
      implicit val aiOrd: Ordering[Any] =
        extendOrderingToNull(missingGreatest = true)(
          new Ordering[Array[Byte]] {
            private val ibord = Ordering.Iterable[Byte]

            def compare(a: Array[Byte], b: Array[Byte]): Int = ibord.compare(a, b)
          })

      def compare(a: Genotype, b: Genotype): Int = {
        val compareGT = a.unboxedGT.compare(b.unboxedGT)
        if (compareGT != 0) return compareGT

        val compareAD = aiOrd.compare(a.unboxedAD, b.unboxedAD)
        if (compareAD != 0) return compareAD

        val compareDP = a.unboxedDP.compare(b.unboxedDP)
        if (compareDP != 0) return compareDP

        val compareGQ = a.unboxedGQ.compare(b.unboxedGQ)
        if (compareGQ != 0) return compareGQ

        val comparePX = aiOrd.compare(a.unboxedPX, b.unboxedPX)
        if (comparePX != 0) return comparePX

        val compareFakeRef = a.fakeRef.compare(b.fakeRef)
        if (compareFakeRef != 0) return compareFakeRef

        a.isLinearScale.compare(b.isLinearScale)
      }
    }
}

class GenericGenotype(val unboxedGT: Int,
  val unboxedAD: Array[Int],
  val unboxedDP: Int,
  val unboxedGQ: Int,
  val unboxedPX: Array[Int],
  val fakeRef: Boolean,
  val isLinearScale: Boolean) extends Genotype {

  require(unboxedGT >= -1, s"invalid _gt value: $unboxedGT")
  require(unboxedDP >= -1, s"invalid _dp value: $unboxedDP")

  if (isLinearScale) {
    require(unboxedGQ == -1)
    if (unboxedPX == null)
      require(unboxedGT == -1)
    else {
      require(unboxedPX.sum == 32768)
      require(unboxedGT == Genotype.gtFromLinear(unboxedPX).getOrElse(-1))
    }
  }
}

class MutableGenotype(nAlleles: Int) extends Genotype {
  var unboxedGT: Int = -1
  private val _ad: Array[Int] = Array.ofDim[Int](nAlleles)
  private var _hasAD = false
  var unboxedDP: Int = -1
  var unboxedGQ: Int = -1
  private val _px: Array[Int] = Array.ofDim[Int](triangle(nAlleles))
  private var _hasPX = false
  var fakeRef: Boolean = false
  var isLinearScale: Boolean = false

  def unboxedAD: Array[Int] = if (_hasAD) _ad else null

  def unboxedPX: Array[Int] = if (_hasPX) _px else null

  def read(nAlleles: Int, isLinearScale: Boolean, a: ByteIterator) {
    val isBiallelic = nAlleles == 2

    val flags = a.readULEB128()

    _hasAD = Genotype.flagHasAD(flags)
    _hasPX = Genotype.flagHasPX(flags)

    unboxedGT =
      if (Genotype.flagHasGT(isBiallelic, flags)) {
        if (Genotype.flagStoresGT(isBiallelic, flags))
          Genotype.flagGT(isBiallelic, flags)
        else
          a.readULEB128()
      } else
        -1

    if (_hasAD) {
      if (Genotype.flagSimpleAD(flags)) {
        assert(unboxedGT >= 0)
        val p = Genotype.gtPair(unboxedGT)
        _ad(p.j) = a.readULEB128()
        if (p.j != p.k)
          _ad(p.k) = a.readULEB128()
      } else {
        for (i <- _ad.indices)
          _ad(i) = a.readULEB128()
      }
    }

    unboxedDP =
      if (Genotype.flagHasDP(flags)) {
        if (_hasAD) {
          var i = 0
          var adsum = 0
          while (i < _ad.length) {
            adsum += _ad(i)
            i += 1
          }
          if (Genotype.flagSimpleDP(flags))
            adsum
          else
            adsum + a.readULEB128()
        } else
          a.readULEB128()
      } else
        -1

    if (_hasPX) {
      if (unboxedGT >= 0) {
        var i = 0
        while (i < unboxedGT) {
          _px(i) = a.readULEB128()
          i += 1
        }
        i += 1
        while (i < _px.length) {
          _px(i) = a.readULEB128()
          i += 1
        }
        if (isLinearScale)
          _px(unboxedGT) = 32768 - _px.sum // original values summed to 32768 or 1.0 in probability
        else
          _px(unboxedGT) = 0
      } else {
        var i = 0
        while (i < _px.length) {
          _px(i) = a.readULEB128()
          i += 1
        }
      }
    }

    unboxedGQ =
      if (Genotype.flagHasGQ(flags)) {
        if (Genotype.flagSimpleGQ(flags))
          Genotype.gqFromPL(_px)
        else
          a.readULEB128()
      } else
        -1

    fakeRef = Genotype.flagFakeRef(flags)

    this.isLinearScale = isLinearScale
  }
}

class DosageGenotype(var unboxedGT: Int,
  var unboxedPX: Array[Int]) extends Genotype {

  def unboxedAD: Array[Int] = null

  def unboxedDP: Int = -1

  def unboxedGQ: Int = -1

  def fakeRef: Boolean = false

  def isLinearScale = true
}

class GenotypeBuilder(nAlleles: Int, isLinearScale: Boolean = false) {
  require(nAlleles > 0, s"tried to create genotype builder with $nAlleles ${ plural(nAlleles, "allele") }")
  val isBiallelic = nAlleles == 2
  val nGenotypes = triangle(nAlleles)

  var flags: Int = 0

  private var gt: Int = 0
  private var ad: Array[Int] = _
  private var dp: Int = 0
  private var gq: Int = 0
  private var px: Array[Int] = _

  def clear() {
    flags = 0
  }

  def hasGT: Boolean =
    Genotype.flagHasGT(isBiallelic, flags)

  def setGT(newGT: Int) {
    if (newGT < 0)
      fatal(s"invalid GT value `$newGT': negative value")
    if (newGT > nGenotypes)
      fatal(s"invalid GT value `$newGT': value larger than maximum number of genotypes $nGenotypes")
    if (hasGT)
      fatal(s"invalid GT, genotype already had GT")
    flags = Genotype.flagSetGT(isBiallelic, flags, newGT)
    gt = newGT
  }

  def setAD(newAD: Array[Int]) {
    if (newAD.length != nAlleles)
      fatal(s"invalid AD field `${ newAD.mkString(",") }': expected $nAlleles values, but got ${ newAD.length }.")
    flags = Genotype.flagSetHasAD(flags)
    ad = newAD
  }

  def setDP(newDP: Int) {
    if (newDP < 0)
      fatal(s"invalid DP field `$newDP': negative value")
    flags = Genotype.flagSetHasDP(flags)
    dp = newDP
  }

  def setGQ(newGQ: Int) {
    if (newGQ < 0)
      fatal(s"invalid GQ field `$newGQ': negative value")
    flags = Genotype.flagSetHasGQ(flags)
    gq = newGQ
  }

  def setPX(newPX: Array[Int]) {
    if (newPX.length != nGenotypes)
      fatal(s"invalid PL field `${ newPX.mkString(",") }': expected $nGenotypes values, but got ${ newPX.length }.")
    flags = Genotype.flagSetHasPX(flags)
    px = newPX
  }

  def setFakeRef() {
    flags = Genotype.flagSetFakeRef(flags)
  }

  def set(g: Genotype) {
    g.gt.foreach(setGT)
    g.ad.foreach(setAD)
    g.dp.foreach(setDP)
    g.gq.foreach(setGQ)
    g.px.foreach(setPX)

    if (g.fakeRef)
      setFakeRef()
  }

  def write(b: ArrayBuilder[Byte]) {
    val hasGT = Genotype.flagHasGT(isBiallelic, flags)
    val hasAD = Genotype.flagHasAD(flags)
    val hasDP = Genotype.flagHasDP(flags)
    val hasGQ = Genotype.flagHasGQ(flags)
    val hasPX = Genotype.flagHasPX(flags)

    if (isLinearScale) {
      if (hasPX) {
        Genotype.gtFromLinear(px) match {
          case Some(gt2) => assert(hasGT && gt == gt2)
          case None => assert(!hasGT)
        }
      } else
        assert(!hasGT)
    }

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

    if (hasPX && hasGQ) {
      val gqFromPL = Genotype.gqFromPL(px)
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

    if (hasPX) {
      if (hasGT) {
        var i = 0
        while (i < gt) {
          b.writeULEB128(px(i))
          i += 1
        }
        i += 1
        while (i < px.length) {
          b.writeULEB128(px(i))
          i += 1
        }
      } else
        px.foreach(b.writeULEB128)
    }

    if (hasGQ) {
      if (!Genotype.flagSimpleGQ(flags))
        b.writeULEB128(gq)
    }
  }
}



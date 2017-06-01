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

  def _unboxedGT: Int

  def _unboxedAD: Array[Int]

  def _unboxedDP: Int

  def _unboxedGQ: Int

  def _unboxedPX: Array[Int]

  def _fakeRef: Boolean

  def _isLinearScale: Boolean

  def check(nAlleles: Int) {
    val nGenotypes = triangle(nAlleles)
    assert(Genotype.gt(this).forall(i => i >= 0 && i < nGenotypes))
    assert(Genotype.ad(this).forall(a => a.length == nAlleles))
    assert(Genotype.px(this).forall(a => a.length == nGenotypes))
  }

  def copy(gt: Option[Int] = Genotype.gt(this),
    ad: Option[Array[Int]] = Genotype.ad(this),
    dp: Option[Int] = Genotype.dp(this),
    gq: Option[Int] = Genotype.gq(this),
    px: Option[Array[Int]] = Genotype.px(this),
    fakeRef: Boolean = this._fakeRef,
    isLinearScale: Boolean = this._isLinearScale): Genotype = Genotype(gt, ad, dp, gq, px, fakeRef, isLinearScale)

  override def equals(that: Any): Boolean = that match {
    case g: Genotype =>
      _unboxedGT == g._unboxedGT &&
        util.Arrays.equals(_unboxedAD, g._unboxedAD) &&
        _unboxedDP == g._unboxedDP &&
        _unboxedGQ == g._unboxedGQ &&
        util.Arrays.equals(_unboxedPX, g._unboxedPX) &&
        _fakeRef == g._fakeRef &&
        _isLinearScale == g._isLinearScale

    case _ => false
  }

  override def hashCode: Int =
    new HashCodeBuilder(43, 19)
      .append(_unboxedGT)
      .append(util.Arrays.hashCode(_unboxedAD))
      .append(_unboxedDP)
      .append(_unboxedGQ)
      .append(util.Arrays.hashCode(_unboxedPX))
      .append(_fakeRef)
      .append(_isLinearScale)
      .toHashCode

  override def toString: String = {
    val b = new StringBuilder

    b.append(Genotype.gt(this).map { gt =>
      val p = Genotype.gtPair(gt)
      s"${ p.j }/${ p.k }"
    }.getOrElse("./."))

    if (_fakeRef) {
      b += '*'
    }

    b += ':'
    b.append(Genotype.ad(this).map(_.mkString(",")).getOrElse("."))
    b += ':'
    b.append(Genotype.dp(this).map(_.toString).getOrElse("."))
    b += ':'
    b.append(Genotype.gq(this).map(_.toString).getOrElse("."))
    b += ':'
    if (!_isLinearScale) {
      b.append("PL=" + Genotype.pl(this).map(_.mkString(",")).getOrElse("."))
    } else {
      b.append("GP=" + Genotype.gp(this).map(_.mkString(",")).getOrElse("."))
    }

    b.result()
  }
}

class RowGenotype(r: Row) extends Genotype {
  def _unboxedGT: Int =
    if (r.isNullAt(0))
      -1
    else
      r.getInt(0)

  def _unboxedAD: Array[Int] =
    if (r.isNullAt(1))
      null
    else
      r.getSeq(1).toArray

  def _unboxedDP: Int =
    if (r.isNullAt(2))
      -1
    else
      r.getInt(2)

  def _unboxedGQ: Int =
    if (r.isNullAt(3))
      -1
    else
      r.getInt(3)

  def _unboxedPX: Array[Int] =
    if (r.isNullAt(4))
      null
    else
      r.getSeq(4).toArray

  // not nullable
  def _fakeRef: Boolean = r.getBoolean(5)

  def _isLinearScale: Boolean = r.getBoolean(6)
}

object Genotype {
  def unboxedGT(g: Genotype): Int = if (g != null) g._unboxedGT else -1

  def unboxedAD(g: Genotype): Array[Int] = if (g != null) g._unboxedAD else null

  def unboxedDP(g: Genotype): Int = if (g != null) g._unboxedDP else -1

  def unboxedGQ(g: Genotype): Int = if (g != null) g._unboxedGQ else -1

  def unboxedPX(g: Genotype): Array[Int] = if (g != null) g._unboxedPX else null

  def unboxedPL(g: Genotype): Array[Int] =
    if (g == null)
      null
    else {
      val upx = g._unboxedPX
      if (upx == null)
        null
      else if (g._isLinearScale)
        Genotype.linearToPhred(upx)
      else
        upx
    }

  def unboxedGP(g: Genotype): Array[Double] = {
    val upx = unboxedPX(g)
    if (upx == null)
      null
    else if (g._isLinearScale)
      upx.map(_ * Genotype.gpNorm)
    else
      Genotype.plToGP(upx)
  }

  def gt(g: Genotype): Option[Int] = if (unboxedGT(g) == -1) None else Some(unboxedGT(g))

  def ad(g: Genotype): Option[Array[Int]] = Option(unboxedAD(g))

  def gq(g: Genotype): Option[Int] = {
    val ugq = unboxedGQ(g)
    if (ugq == -1)
      None
    else
      Some(ugq)
  }

  def pl(g: Genotype): Option[Array[Int]] = Option(unboxedPL(g))

  def dp(g: Genotype): Option[Int] = {
    val udp = unboxedDP(g)
    if (udp == -1)
      None
    else
      Some(udp)
  }

  def gp(g: Genotype): Option[Array[Double]] = Option(unboxedGP(g))

  def px(g: Genotype): Option[Array[Int]] = Option(unboxedPX(g))

  def unboxedDosage(g: Genotype): Double = {
    val upx = unboxedPX(g)
    if (upx == null)
      -1d
    else if (upx.size != 3)
      fatal("Genotype dosage is not defined for multi-allelic variants")
    else if (g._isLinearScale)
      (upx(1) + 2 * upx(2)) * Genotype.gpNorm
    else
      Genotype.plToDosage(upx(0), upx(1), upx(2))
  }

  def dosage(g: Genotype): Option[Double] = {
    val ud = unboxedDosage(g)
    if (ud == -1d)
      None
    else
      Some(ud)
  }

  def call(g: Genotype): Call = {
    val ugt = unboxedGT(g)
    if (ugt == -1)
      null
    else
      Call(box(ugt))
  }

  def isHomRef(g: Genotype): Boolean = unboxedGT(g) == 0

  def isHet(g: Genotype): Boolean = unboxedGT(g) > 0 && {
    val p = Genotype.gtPair(unboxedGT(g))
    p.j != p.k
  }

  def isHomVar(g: Genotype): Boolean = unboxedGT(g) > 0 && {
    val p = Genotype.gtPair(unboxedGT(g))
    p.j == p.k
  }

  def isCalledNonRef(g: Genotype): Boolean = unboxedGT(g) > 0

  def isHetNonRef(g: Genotype): Boolean = unboxedGT(g) > 0 && {
    val p = Genotype.gtPair(unboxedGT(g))
    p.j > 0 && p.j != p.k
  }

  def isHetRef(g: Genotype): Boolean = unboxedGT(g) > 0 && {
    val p = Genotype.gtPair(unboxedGT(g))
    p.j == 0 && p.k > 0
  }

  def gtType(g: Genotype): GenotypeType =
    if (isHomRef(g))
      GenotypeType.HomRef
    else if (isHet(g))
      GenotypeType.Het
    else if (isHomVar(g))
      GenotypeType.HomVar
    else {
      assert(!Genotype.isCalled(g))
      GenotypeType.NoCall
    }

  def isCalled(g: Genotype): Boolean = unboxedGT(g) >= 0

  def isNotCalled(g: Genotype): Boolean = unboxedGT(g) == -1

  def hasOD(g: Genotype): Boolean = unboxedDP(g) != -1 && unboxedAD(g) != null

  def od_(g: Genotype): Int = unboxedDP(g) - intArraySum(unboxedAD(g))

  def od(g: Genotype): Option[Int] =
    if (hasOD(g))
      Some(od_(g))
    else
      None

  def hasNNonRefAlleles(g: Genotype): Boolean = unboxedGT(g) != -1

  def nNonRefAlleles_(g: Genotype): Int = Genotype.gtPair(unboxedGT(g)).nNonRefAlleles

  def nNonRefAlleles(g: Genotype): Option[Int] =
    if (hasNNonRefAlleles(g))
      Some(nNonRefAlleles_(g))
    else
      None

  def fakeRef(g: Genotype): Option[Boolean] = if (g == null) None else Some(g._fakeRef)

  def isLinearScale(g: Genotype): Option[Boolean] = if (g == null) None else Some(g._isLinearScale)

  def fractionReadsRef(g: Genotype): Option[Double] = {
    val uad = unboxedAD(g)
    if (uad != null) {
      val s = intArraySum(uad)
      if (s != 0)
        Some(uad(0).toDouble / s)
      else
        None
    } else
      None
  }

  def oneHotAlleles(nAlleles: Int, g: Genotype): Option[IndexedSeq[Int]] = {
    Genotype.gt(g).map { call =>
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

  def oneHotAlleles(v: Variant, g: Genotype): Option[IndexedSeq[Int]] = oneHotAlleles(v.nAlleles, g)

  def oneHotGenotype(v: Variant, g: Genotype): Option[IndexedSeq[Int]] = oneHotGenotype(v.nGenotypes, g)

  def oneHotGenotype(nGenotypes: Int, g: Genotype): Option[IndexedSeq[Int]] = {
    Genotype.gt(g).map { call =>
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


  def hasPAB(g: Genotype): Boolean = unboxedAD(g) != null && Genotype.isHet(g)

  def pAB_(g: Genotype, theta: Double = 0.5): Double = {
    val gtPair = Genotype.gtPair(unboxedGT(g))
    val uad = unboxedAD(g)
    val aDepth = uad(gtPair.j)
    val bDepth = uad(gtPair.k)
    val d = new BinomialDistribution(aDepth + bDepth, theta)
    val minDepth = aDepth.min(bDepth)
    val minp = d.probability(minDepth)
    val mincp = d.cumulativeProbability(minDepth)
    (2 * mincp - minp).min(1.0).max(0.0)
  }

  def pAB(g: Genotype, theta: Double = 0.5): Option[Double] =
    if (hasPAB(g))
      Some(pAB_(g, theta))
    else
      None

  def apply(gtx: Int): Genotype = new GenericGenotype(gtx, null, -1, -1, null, false, false)

  def toRow(g: Genotype): Row =
    if (g == null)
      null
    else {
      val gt = unboxedGT(g)
      val dp = unboxedDP(g)
      val gq = unboxedGQ(g)
      Row(
        if (gt == -1) null else gt,
        unboxedAD(g): IndexedSeq[Int],
        if (dp == -1) null else dp,
        if (gq == -1) null else gq,
        unboxedPX(g): IndexedSeq[Int],
        g._fakeRef,
        g._isLinearScale)
    }

  def toJSON(g: Genotype): JValue =
    if (g == null)
      JNull
    else
      JObject(
        ("gt", Genotype.gt(g).map(JInt(_)).getOrElse(JNull)),
        ("ad", Genotype.ad(g).map(ads => JArray(ads.map(JInt(_)).toList)).getOrElse(JNull)),
        ("dp", Genotype.dp(g).map(JInt(_)).getOrElse(JNull)),
        ("gq", Genotype.gq(g).map(JInt(_)).getOrElse(JNull)),
        ("px", Genotype.px(g).map(pxs => JArray(pxs.map(JInt(_)).toList)).getOrElse(JNull)),
        ("fakeRef", JBool(g._fakeRef)),
        ("isLinearScale", JBool(g._isLinearScale)))

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

    val g = new GenericGenotype(gtx, ad, dpx, gqx, pl, _fakeRef = false, _isLinearScale = false)
    g.check(nAlleles)
    g
  }

  def apply(unboxedGT: Int, fakeRef: Boolean): Genotype =
    new GenericGenotype(unboxedGT, null, -1, -1, null, fakeRef, false)

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
  final val flagMissing = 0x400

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

  def flagMissing(flags: Int): Boolean = (flags & flagMissing) != 0

  def flagSetFakeRef(flags: Int): Int = flags | flagFakeRefBit

  def flagSetMissing(flags: Int): Int = flags | flagMissing

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
    for (i <- 0 until n) {
      aAcc += a(i).toDouble
      val t = (aAcc * 32768 / s + 0.5).toInt
      r(i) = t - rAcc
      rAcc = t
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
    for (i <- 0 until n) {
      aAcc += a(i).toDouble
      val t = (aAcc * 32768 / s + 0.5).toInt
      r(i) = t - rAcc
      rAcc = t
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
    if (flagMissing(flags))
      return null

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
    val gg = for (px <- Gen.option(Gen.partition(nGenotypes, 32768))) yield {
      val gt = px.flatMap(gtFromLinear)
      val g = Genotype(gt = gt, px = px, isLinearScale = true)
      g.check(nAlleles)
      g
    }
    Gen.frequency(
      (100, gg),
      (1, Gen.const(null)))
  }

  def genExtreme(v: Variant): Gen[Genotype] = {
    val nAlleles = v.nAlleles
    val m = Int.MaxValue / (nAlleles + 1)
    val nGenotypes = triangle(nAlleles)
    val gg = for (gt: Option[Int] <- Gen.option(Gen.choose(0, nGenotypes - 1));
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
    Gen.frequency(
      (100, gg),
      (1, Gen.const(null)))
  }

  def genRealistic(v: Variant): Gen[Genotype] = {
    val nAlleles = v.nAlleles
    val nGenotypes = triangle(nAlleles)
    val gg = for (callRate <- Gen.choose(0d, 1d);
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

    Gen.frequency(
      (100, gg),
      (1, Gen.const(null)))
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
    extendOrderingToNull[Genotype](missingGreatest = true)(
      new Ordering[Genotype] {
        implicit val aiOrd: Ordering[Array[Int]] =
          extendOrderingToNull(missingGreatest = true)(
            new Ordering[Array[Int]] {
              private val ibord = Ordering.Iterable[Int]

              def compare(a: Array[Int], b: Array[Int]): Int = ibord.compare(a, b)
            })

        def compare(a: Genotype, b: Genotype): Int = {
          val compareGT = a._unboxedGT.compare(b._unboxedGT)
          if (compareGT != 0) return compareGT

          val compareAD = aiOrd.compare(a._unboxedAD, b._unboxedAD)
          if (compareAD != 0) return compareAD

          val compareDP = a._unboxedDP.compare(b._unboxedDP)
          if (compareDP != 0) return compareDP

          val compareGQ = a._unboxedGQ.compare(b._unboxedGQ)
          if (compareGQ != 0) return compareGQ

          val comparePX = aiOrd.compare(a._unboxedPX, b._unboxedPX)
          if (comparePX != 0) return comparePX

          val compareFakeRef = a._fakeRef.compare(b._fakeRef)
          if (compareFakeRef != 0) return compareFakeRef

          a._isLinearScale.compare(b._isLinearScale)
        }
      })
}

class GenericGenotype(val _unboxedGT: Int,
  val _unboxedAD: Array[Int],
  val _unboxedDP: Int,
  val _unboxedGQ: Int,
  val _unboxedPX: Array[Int],
  val _fakeRef: Boolean,
  val _isLinearScale: Boolean) extends Genotype {

  require(_unboxedGT >= -1, s"invalid _unboxedGT value: ${ _unboxedGT }")
  require(_unboxedDP >= -1, s"invalid _unboxedDP value: ${ _unboxedDP }")

  if (_isLinearScale) {
    require(_unboxedGQ == -1)
    if (_unboxedPX == null)
      require(_unboxedGT == -1)
    else {
      require(_unboxedPX.sum == 32768)
      require(_unboxedGT == Genotype.gtFromLinear(_unboxedPX).getOrElse(-1))
    }
  }
}

class MutableGenotype(nAlleles: Int) extends Genotype {
  var _unboxedGT: Int = -1
  private val _ad: Array[Int] = Array.ofDim[Int](nAlleles)
  private var _hasAD = false
  var _unboxedDP: Int = -1
  var _unboxedGQ: Int = -1
  private val _px: Array[Int] = Array.ofDim[Int](triangle(nAlleles))
  private var _hasPX = false
  var _fakeRef: Boolean = false
  var _isLinearScale: Boolean = false

  def _unboxedAD: Array[Int] = if (_hasAD) _ad else null

  def _unboxedPX: Array[Int] = if (_hasPX) _px else null

  def read(nAlleles: Int, isLinearScale: Boolean, a: ByteIterator): Boolean = {
    val isBiallelic = nAlleles == 2

    val flags = a.readULEB128()

    val missing = Genotype.flagMissing(flags)
    if (missing)
      return false

    _hasAD = Genotype.flagHasAD(flags)
    _hasPX = Genotype.flagHasPX(flags)

    _unboxedGT =
      if (Genotype.flagHasGT(isBiallelic, flags)) {
        if (Genotype.flagStoresGT(isBiallelic, flags))
          Genotype.flagGT(isBiallelic, flags)
        else
          a.readULEB128()
      } else
        -1

    if (_hasAD) {
      if (Genotype.flagSimpleAD(flags)) {
        assert(_unboxedGT >= 0)
        val p = Genotype.gtPair(_unboxedGT)
        _ad(p.j) = a.readULEB128()
        if (p.j != p.k)
          _ad(p.k) = a.readULEB128()
      } else {
        for (i <- _ad.indices)
          _ad(i) = a.readULEB128()
      }
    }

    _unboxedDP =
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
      if (_unboxedGT >= 0) {
        var i = 0
        while (i < _unboxedGT) {
          _px(i) = a.readULEB128()
          i += 1
        }
        i += 1
        while (i < _px.length) {
          _px(i) = a.readULEB128()
          i += 1
        }
        if (isLinearScale)
          _px(_unboxedGT) = 32768 - _px.sum // original values summed to 32768 or 1.0 in probability
        else
          _px(_unboxedGT) = 0
      } else {
        var i = 0
        while (i < _px.length) {
          _px(i) = a.readULEB128()
          i += 1
        }
      }
    }

    _unboxedGQ =
      if (Genotype.flagHasGQ(flags)) {
        if (Genotype.flagSimpleGQ(flags))
          Genotype.gqFromPL(_px)
        else
          a.readULEB128()
      } else
        -1

    _fakeRef = Genotype.flagFakeRef(flags)

    this._isLinearScale = isLinearScale

    true
  }
}

class DosageGenotype(var _unboxedGT: Int,
  var _unboxedPX: Array[Int]) extends Genotype {

  def _unboxedAD: Array[Int] = null

  def _unboxedDP: Int = -1

  def _unboxedGQ: Int = -1

  def _fakeRef: Boolean = false

  def _isLinearScale = true
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

  def setMissing() {
    flags = Genotype.flagSetMissing(flags)
  }

  def set(g: Genotype) {
    if (g == null) {
      setMissing()
    } else {
      Genotype.gt(g).foreach(setGT)
      Genotype.ad(g).foreach(setAD)
      Genotype.dp(g).foreach(setDP)
      Genotype.gq(g).foreach(setGQ)
      Genotype.px(g).foreach(setPX)

      if (g._fakeRef)
        setFakeRef()
    }
  }

  def write(b: ArrayBuilder[Byte]) {
    val missing = Genotype.flagMissing(flags)
    if (missing) {
      b.writeULEB128(flags)
      return
    }

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



package is.hail.variant

import java.util

import is.hail.annotations.Annotation
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr._
import is.hail.expr.typ.{TArray, TStruct}
import is.hail.utils._
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.apache.spark.sql.Row

import scala.language.implicitConversions

object GenotypeType extends Enumeration {
  type GenotypeType = Value
  val HomRef = Value(0)
  val Het = Value(1)
  val HomVar = Value(2)
  val NoCall = Value(-1)
}

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


object Genotype {
  val htsGenotypeType: TStruct = TStruct(
    "GT" -> TCall(),
    "AD" -> TArray(!TInt32()),
    "DP" -> TInt32(),
    "GQ" -> TInt32(),
    "PL" -> TArray(!TInt32()))

  def unboxedGT(a: Annotation): Int = {
    if (a == null)
      -1
    else {
      val r = a.asInstanceOf[Row]
      if (r.isNullAt(0))
        -1
      else
        r.getInt(0)
    }
  }

  def unboxedAD(g: Annotation): Array[Int] = {
    if (g != null) {
      val r = g.asInstanceOf[Row]
      if (r.isNullAt(1))
        null
      else
        r.getAs[IndexedSeq[Int]](1).toArray
    } else
      null
  }

  def unboxedDP(g: Annotation): Int = {
    if (g == null)
      -1
    else {
      val r = g.asInstanceOf[Row]
      if (r.isNullAt(2))
        -1
      else
        r.getInt(2)
    }
  }

  def unboxedGQ(g: Annotation): Int = {
    if (g == null)
      -1
    else {
      val r = g.asInstanceOf[Row]
      if (r.isNullAt(3))
        -1
      else
        r.getInt(3)
    }
  }

  def unboxedPL(g: Annotation): Array[Int] = {
    if (g != null) {
      val r = g.asInstanceOf[Row]
      if (r.isNullAt(4))
        null
      else
        r.getAs[IndexedSeq[Int]](4).toArray
    } else
      null
  }

  def gt(g: Annotation): Option[Int] = if (unboxedGT(g) == -1) None else Some(unboxedGT(g))

  def ad(g: Annotation): Option[Array[Int]] = Option(unboxedAD(g))

  def gq(g: Annotation): Option[Int] = {
    val ugq = unboxedGQ(g)
    if (ugq == -1)
      None
    else
      Some(ugq)
  }

  def pl(g: Annotation): Option[Array[Int]] = Option(unboxedPL(g))

  def dp(g: Annotation): Option[Int] = {
    val udp = unboxedDP(g)
    if (udp == -1)
      None
    else
      Some(udp)
  }

  def apply(gtx: Int): Annotation = Annotation(if (gtx == -1) null else gtx, null, null, null, null)

  def apply(gt: java.lang.Integer, ad: Array[Int], dp: java.lang.Integer, gq: java.lang.Integer, pl: Array[Int]): Annotation =
    Annotation(gt, ad: IndexedSeq[Int], dp, gq, pl: IndexedSeq[Int])

  def apply(gt: Option[Int] = None,
    ad: Option[Array[Int]] = None,
    dp: Option[Int] = None,
    gq: Option[Int] = None,
    pl: Option[Array[Int]] = None): Annotation =
    Annotation(gt.orNull, ad.map(adx => adx: IndexedSeq[Int]).orNull, dp.orNull, gq.orNull, pl.map(plx => plx: IndexedSeq[Int]).orNull)

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
    assert(m <= m2)
    m2 - m
  }

  def gtFromPL(a: IndexedSeq[Int]): Call = {
    def f(i: Int, m: Int, mi: Int, count: Int): Call = {
      if (i == a.length) {
        assert(count >= 1)
        if (count == 1)
          mi
        else
          null
      } else if (a(i) < m)
        f(i + 1, a(i), i, 1)
      else if (a(i) == m)
        f(i + 1, m, mi, count + 1)
      else
        f(i + 1, m, mi, count)
    }

    f(1, a(0), 0, 1)
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

  def unboxedGTFromLinear(a: Array[Double]): Int = {
    def f(i: Int, m: Double, mi: Int, count: Int): Int = {
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

  def unboxedGTFromUIntLinear(a: ArrayUInt): Int = {
    def f(i: Int, m: UInt, mi: Int, count: Int): Int = {
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

  val maxPhredInTable = 8192

  lazy val phredToLinearConversionTable: Array[Double] = (0 to maxPhredInTable).map { i => math.pow(10, i / -10.0) }.toArray

  def phredToLinear(i: Int): Double =
    if (i < maxPhredInTable) phredToLinearConversionTable(i) else math.pow(10, i / -10.0)

  def plToDosage(pl0: Int, pl1: Int, pl2: Int): Double = {
    val p0 = phredToLinear(pl0)
    val p1 = phredToLinear(pl1)
    val p2 = phredToLinear(pl2)

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

  def genExtremeNonmissing(v: Variant): Gen[Annotation] = {
    val nAlleles = v.nAlleles
    val m = Int.MaxValue / (nAlleles + 1)
    val nGenotypes = triangle(nAlleles)
    val gg = for (gt: Option[Int] <- Gen.option(Gen.choose(0, nGenotypes - 1));
      ad <- Gen.option(Gen.buildableOfN[Array](nAlleles, Gen.choose(0, m)));
      dp <- Gen.option(Gen.choose(0, m));
      gq <- Gen.option(Gen.choose(0, 10000));
      pl <- Gen.oneOfGen(
        Gen.option(Gen.buildableOfN[Array](nGenotypes, Gen.choose(0, m))),
        Gen.option(Gen.buildableOfN[Array](nGenotypes, Gen.choose(0, 100))))) yield {
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
      val g = Annotation(gt.orNull,
        ad.map(a => a: IndexedSeq[Int]).orNull,
        dp.map(_ + ad.map(_.sum).getOrElse(0)).orNull,
        gq.orNull,
        pl.map(a => a: IndexedSeq[Int]).orNull)
      g
    }
    gg
  }

  def genExtreme(v: Variant): Gen[Annotation] = {
    Gen.frequency(
      (100, genExtremeNonmissing(v)),
      (1, Gen.const(null)))
  }

  def genRealisticNonmissing(v: Variant): Gen[Annotation] = {
    val nAlleles = v.nAlleles
    val nGenotypes = triangle(nAlleles)
    val gg = for (callRate <- Gen.choose(0d, 1d);
      alleleFrequencies <- Gen.buildableOfN[Array](nAlleles, Gen.choose(1e-6, 1d)) // avoid divison by 0
        .map { rawWeights =>
        val sum = rawWeights.sum
        rawWeights.map(_ / sum)
      };
      gt <- Gen.option(Gen.zip(Gen.chooseWithWeights(alleleFrequencies), Gen.chooseWithWeights(alleleFrequencies))
        .map { case (gti, gtj) => gtIndexWithSwap(gti, gtj) }, callRate);
      ad <- Gen.option(Gen.buildableOfN[Array](nAlleles,
        Gen.choose(0, 50)));
      dp <- Gen.choose(0, 30).map(d => ad.map(o => o.sum + d));
      pl <- Gen.option(Gen.buildableOfN[Array](nGenotypes, Gen.choose(0, 1000)).map { arr =>
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
      Annotation(gt.orNull, ad.map(a => a: IndexedSeq[Int]).orNull, dp.orNull, gq.orNull, pl.map(a => a: IndexedSeq[Int]).orNull)
    gg
  }

  def genRealistic(v: Variant): Gen[Annotation] = {
    Gen.frequency(
      (100, genRealisticNonmissing(v)),
      (1, Gen.const(null)))
  }


  def genGenericDosageGenotype(v: Variant): Gen[Annotation] = {
    val nAlleles = v.nAlleles
    val nGenotypes = triangle(nAlleles)
    val gg = for (gp <- Gen.option(Gen.partition(nGenotypes, 32768))) yield {
      val gt = gp.flatMap(gtFromLinear)
      Row(
        gt.orNull,
        gp.map(gpx => gpx.map(p => p.toDouble / 32768): IndexedSeq[Double]).orNull)
    }
    Gen.frequency(
      (100, gg),
      (1, Gen.const(null)))
  }

  def genVariantGenotype: Gen[(Variant, Annotation)] =
    for (v <- Variant.gen;
      g <- Gen.oneOfGen(genExtreme(v), genRealistic(v)))
      yield (v, g)

  def genNonmissingValue: Gen[Annotation] = Variant.gen.flatMap(v => Gen.oneOfGen(genExtremeNonmissing(v), genRealisticNonmissing(v)))

  def genArb: Gen[Annotation] = Variant.gen.flatMap(v => Gen.oneOfGen(genExtreme(v), genRealistic(v)))

  implicit def arbGenotype = Arbitrary(genArb)
}

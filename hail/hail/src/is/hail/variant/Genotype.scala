package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.types.virtual.{TArray, TCall, TInt32, TStruct}

import org.apache.spark.sql.Row

object GenotypeType extends Enumeration {
  type GenotypeType = Value
  val HomRef = Value(0)
  val Het = Value(1)
  val HomVar = Value(2)
  val NoCall = Value(-1)
}

object AllelePair {
  def apply(j: Int, k: Int): Int = {
    require(j >= 0 && j <= 0xffff, s"GTPair invalid j value $j")
    require(k >= 0 && k <= 0xffff, s"GTPair invalid k value $k")
    j | (k << 16)
  }

  def fromNonNormalized(j: Int, k: Int): Int =
    if (j <= k)
      AllelePair(j, k)
    else
      AllelePair(k, j)

  def j(p: Int): Int = p & 0xffff
  def k(p: Int): Int = (p >> 16) & 0xffff

  def nNonRefAlleles(p: Int): Int =
    (if (j(p) != 0) 1 else 0) + (if (k(p) != 0) 1 else 0)

  def alleleIndices(p: Int): Array[Int] = Array(j(p), k(p))
}

object Genotype {
  val htsGenotypeType: TStruct = TStruct(
    "GT" -> TCall,
    "AD" -> TArray(TInt32),
    "DP" -> TInt32,
    "GQ" -> TInt32,
    "PL" -> TArray(TInt32),
  )

  def call(g: Annotation): Option[Call] = {
    if (g == null)
      None
    else {
      val r = g.asInstanceOf[Row]
      if (r.isNullAt(0))
        None
      else
        Some(r.getInt(0))
    }
  }

  def apply(c: BoxedCall): Annotation = Annotation(c, null, null, null, null)

  def apply(
    c: BoxedCall,
    ad: Array[Int],
    dp: java.lang.Integer,
    gq: java.lang.Integer,
    pl: Array[Int],
  ): Annotation =
    Annotation(c, ad: IndexedSeq[Int], dp, gq, pl: IndexedSeq[Int])

  def apply(
    c: Option[Call] = None,
    ad: Option[Array[Int]] = None,
    dp: Option[Int] = None,
    gq: Option[Int] = None,
    pl: Option[Array[Int]] = None,
  ): Annotation =
    Annotation(
      c.orNull,
      ad.map(adx => adx: IndexedSeq[Int]).orNull,
      dp.orNull,
      gq.orNull,
      pl.map(plx => plx: IndexedSeq[Int]).orNull,
    )

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

  val maxPhredInTable = 8192

  lazy val phredToLinearConversionTable: Array[Double] = (0 to maxPhredInTable).map { i =>
    math.pow(10, i / -10.0)
  }.toArray

  def phredToLinear(i: Int): Double =
    if (i < maxPhredInTable) phredToLinearConversionTable(i) else math.pow(10, i / -10.0)

  def plToDosage(pl0: Int, pl1: Int, pl2: Int): Double = {
    val p0 = phredToLinear(pl0)
    val p1 = phredToLinear(pl1)
    val p2 = phredToLinear(pl2)

    (p1 + 2 * p2) / (p0 + p1 + p2)
  }

  val smallAllelePair: Array[Int] = Array(
    AllelePair(0, 0),
    AllelePair(0, 1),
    AllelePair(1, 1),
    AllelePair(0, 2),
    AllelePair(1, 2),
    AllelePair(2, 2),
    AllelePair(0, 3),
    AllelePair(1, 3),
    AllelePair(2, 3),
    AllelePair(3, 3),
    AllelePair(0, 4),
    AllelePair(1, 4),
    AllelePair(2, 4),
    AllelePair(3, 4),
    AllelePair(4, 4),
    AllelePair(0, 5),
    AllelePair(1, 5),
    AllelePair(2, 5),
    AllelePair(3, 5),
    AllelePair(4, 5),
    AllelePair(5, 5),
    AllelePair(0, 6),
    AllelePair(1, 6),
    AllelePair(2, 6),
    AllelePair(3, 6),
    AllelePair(4, 6),
    AllelePair(5, 6),
    AllelePair(6, 6),
    AllelePair(0, 7),
    AllelePair(1, 7),
    AllelePair(2, 7),
    AllelePair(3, 7),
    AllelePair(4, 7),
    AllelePair(5, 7),
    AllelePair(6, 7),
    AllelePair(7, 7),
  )

  val smallAlleleJ: Array[Int] = smallAllelePair.map(AllelePair.j)
  val smallAlleleK: Array[Int] = smallAllelePair.map(AllelePair.k)

  val nCachedAllelePairs: Int = smallAllelePair.length

  def cachedAlleleJ(p: Int): Int = smallAlleleJ(p)
  def cachedAlleleK(p: Int): Int = smallAlleleK(p)

  def allelePairRecursive(i: Int): Int = {
    def f(j: Int, k: Int): Int = if (j <= k)
      AllelePair(j, k)
    else
      f(j - k - 1, k + 1)

    f(i, 0)
  }

  def allelePairSqrt(i: Int): Int = {
    val k: Int = (Math.sqrt(8 * i.toDouble + 1) / 2 - 0.5).toInt
    assert(k * (k + 1) / 2 <= i)
    val j = i - k * (k + 1) / 2
    assert(diploidGtIndex(j, k) == i)
    AllelePair(j, k)
  }

  def allelePair(i: Int): Int =
    if (i < smallAllelePair.length)
      smallAllelePair(i)
    else
      allelePairSqrt(i)

  def diploidGtIndex(j: Int, k: Int): Int = {
    if (j < 0 | j > k) {
      throw new AssertionError(s"invalid gtIndex: ($j, $k)")
    }
    k * (k + 1) / 2 + j
  }

  def diploidGtIndex(p: Int): Int = diploidGtIndex(AllelePair.j(p), AllelePair.k(p))

  def diploidGtIndexWithSwap(i: Int, j: Int): Int =
    if (j < i)
      diploidGtIndex(j, i)
    else
      diploidGtIndex(i, j)

}

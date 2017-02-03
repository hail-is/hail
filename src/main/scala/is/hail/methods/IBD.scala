package is.hail.methods

import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.expr.{EvalContext, Parser, TVariant, Type}
import is.hail.variant.{Genotype, Variant, VariantDataset}

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.reflect.ClassTag

object IBDInfo {
  def apply(Z0: Double, Z1: Double, Z2: Double): IBDInfo = {
    IBDInfo(Z0, Z1, Z2, Z1 / 2 + Z2)
  }
}

case class IBDInfo(Z0: Double, Z1: Double, Z2: Double, PI_HAT: Double) {
  def pointwiseMinus(that: IBDInfo): IBDInfo =
    IBDInfo(Z0 - that.Z0, Z1 - that.Z1, Z2 - that.Z2, PI_HAT - that.PI_HAT)

  def hasNaNs: Boolean = Array(Z0, Z1, Z2, PI_HAT).exists(_.isNaN)
}

case class ExtendedIBDInfo(ibd: IBDInfo, ibs0: Long, ibs1: Long, ibs2: Long) {
  def pointwiseMinus(that: ExtendedIBDInfo): ExtendedIBDInfo =
    ExtendedIBDInfo(ibd.pointwiseMinus(that.ibd), ibs0 - that.ibs0, ibs1 - that.ibs1, ibs2 - that.ibs2)

  def hasNaNs: Boolean = ibd.hasNaNs
}

case class IBSExpectations(
  E00: Double, E10: Double, E20: Double,
  E11: Double, E21: Double, E22: Double = 1, nonNaNCount: Int = 1) {
  def hasNaNs: Boolean = Array(E00, E10, E20, E11, E21).exists(_.isNaN)

  def normalized: IBSExpectations =
    IBSExpectations(E00 / nonNaNCount, E10 / nonNaNCount, E20 / nonNaNCount, E11 / nonNaNCount, E21 / nonNaNCount, E22, this.nonNaNCount)

  def scaled(N: Long): IBSExpectations =
    IBSExpectations(E00 * N, E10 * N, E20 * N, E11 * N, E21 * N, E22 * N, this.nonNaNCount)

  def join(that: IBSExpectations): IBSExpectations =
    if (this.hasNaNs)
      that
    else if (that.hasNaNs)
      this
    else
      IBSExpectations(E00 + that.E00,
        E10 + that.E10,
        E20 + that.E20,
        E11 + that.E11,
        E21 + that.E21,
        nonNaNCount = nonNaNCount + that.nonNaNCount)

}

object IBSExpectations {
  def empty: IBSExpectations = IBSExpectations(0, 0, 0, 0, 0, nonNaNCount = 0)
}

object IBD {
  def indicator(b: Boolean): Int = if (b) 1 else 0

  def countRefs(gtIdx: Int): Int = {
    val gt = Genotype.gtPair(gtIdx)
    indicator(gt.j == 0) + indicator(gt.k == 0)
  }

  def ibsForGenotypes(gs: Iterable[Genotype], maybeMaf: Option[Double]): IBSExpectations = {
    def calculateCountsFromMAF(maf: Double) = {
      val Na = gs.count(_.gt.isDefined) * 2.0
      val p = 1 - maf
      val q = maf
      val x = Na * p
      val y = Na * q
      (Na, x, y, p, q)
    }

    def estimateFrequenciesFromSample = {
      val (na, x) = gs.foldLeft((0, 0d)) { case ((na, sum), g) =>
        (na + g.gt.map(x => 2).getOrElse(0), sum + g.gt.map(countRefs).getOrElse(0))
      }
      val Na = na.toDouble
      val y = Na - x
      val p = x / Na
      val q = y / Na
      (Na, x, y, p, q)
    }

    val (na, x, y, p, q) =
      maybeMaf.map(calculateCountsFromMAF).getOrElse(estimateFrequenciesFromSample)
    val Na = na

    val a00 = 2 * p * p * q * q * ((x - 1) / x * (y - 1) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3)))
    val a10 = 4 * p * p * p * q * ((x - 1) / x * (x - 2) / x * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3))) + 4 * p * q * q * q * ((y - 1) / y * (y - 2) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3)))
    val a20 = q * q * q * q * ((y - 1) / y * (y - 2) / y * (y - 3) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3))) + p * p * p * p * ((x - 1) / x * (x - 2) / x * (x - 3) / x * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3))) + 4 * p * p * q * q * ((x - 1) / x * (y - 1) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3)))
    val a11 = 2 * p * p * q * ((x - 1) / x * Na / (Na - 1) * Na / (Na - 2)) + 2 * p * q * q * ((y - 1) / y * Na / (Na - 1) * Na / (Na - 2))
    val a21 = p * p * p * ((x - 1) / x * (x - 2) / x * Na / (Na - 1) * Na / (Na - 2)) + q * q * q * ((y - 1) / y * (y - 2) / y * Na / (Na - 1) * Na / (Na - 2)) + p * p * q * ((x - 1) / x * Na / (Na - 1) * Na / (Na - 2)) + p * q * q * ((y - 1) / y * Na / (Na - 1) * Na / (Na - 2))
    IBSExpectations(a00, a10, a20, a11, a21)
  }

  def calculateIBDInfo(N0: Long, N1: Long, N2: Long, ibse: IBSExpectations, bounded: Boolean): ExtendedIBDInfo = {
    val ibseN = ibse.scaled(N0 + N1 + N2)
    val Z0 = N0 / ibseN.E00
    val Z1 = (N1 - Z0 * ibseN.E10) / ibseN.E11
    val Z2 = (N2 - Z0 * ibseN.E20 - Z1 * ibseN.E21) / ibseN.E22
    val ibd = if (bounded) {
      if (Z0 > 1) {
        IBDInfo(1, 0, 0)
      } else if (Z1 > 1) {
        IBDInfo(0, 1, 0)
      } else if (Z2 > 1) {
        IBDInfo(0, 0, 1)
      } else if (Z0 < 0) {
        val S = Z1 + Z2
        IBDInfo(0, Z1 / S, Z2 / S)
      } else if (Z1 < 0) {
        val S = Z0 + Z2
        IBDInfo(Z0 / S, 0, Z2 / S)
      } else if (Z2 < 0) {
        val S = Z0 + Z1
        IBDInfo(Z0 / S, Z1 / S, 0)
      } else {
        IBDInfo(Z0, Z1, Z2)
      }
    } else {
      IBDInfo(Z0, Z1, Z2)
    }

    ExtendedIBDInfo(ibd, N0, N1, N2)
  }

  final val chunkSize = 1024

  def computeIBDMatrix(vds: VariantDataset, computeMaf: Option[(Variant, Annotation) => Double], bounded: Boolean): RDD[((Int, Int), ExtendedIBDInfo)] = {
    val unnormalizedIbse = vds.rdd.map { case (v, (va, gs)) => ibsForGenotypes(gs, computeMaf.map(f => f(v, va))) }
      .fold(IBSExpectations.empty)(_ join _)

    val ibse = unnormalizedIbse.normalized

    val nSamples = vds.nSamples

    val chunkedGenotypeMatrix = vds.rdd
      .map { case (v, (va, gs)) => gs.map(_.gt.map(IBSFFI.gtToCRep).getOrElse(IBSFFI.missingGTCRep)).toArray[Byte] }
      .zipWithIndex()
      .flatMap { case (gts, variantId) =>
        val vid = (variantId % chunkSize).toInt
        gts.grouped(chunkSize)
          .zipWithIndex
          .map { case (gtGroup, i) => ((i, variantId / chunkSize), (vid, gtGroup)) }
      }
      .aggregateByKey(Array.tabulate(chunkSize * chunkSize)((i) => IBSFFI.missingGTCRep))({ case (x, (vid, gs)) =>
        for (i <- gs.indices) x(vid * chunkSize + i) = gs(i)
        x
      }, { case (x, y) =>
        for (i <- y.indices)
          if (x(i) == IBSFFI.missingGTCRep)
            x(i) = y(i)
        x
      })
      .map { case ((s, v), gs) => (v, (s, IBSFFI.pack(chunkSize, chunkSize, gs))) }

    chunkedGenotypeMatrix.join(chunkedGenotypeMatrix)
      // optimization: Ignore chunks below the diagonal
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (_, ((s1, gs1), (s2, gs2))) =>
        ((s1, s2), IBSFFI.ibs(chunkSize, chunkSize, gs1, gs2))
      }
      .reduceByKey { (a, b) =>
        var i = 0
        while (i != a.length) {
          a(i) += b(i)
          i += 1
        }
        a
      }
      .mapValues { ibs =>
        val arr = new Array[ExtendedIBDInfo](chunkSize * chunkSize)
        var si = 0
        var sj = 0
        while (si != chunkSize * chunkSize) {
          while (sj != chunkSize) {
            arr(si + sj) =
              calculateIBDInfo(ibs(si * 3 + sj * 3), ibs(si * 3 + sj * 3 + 1), ibs(si * 3 + sj * 3 + 2), ibse, bounded)
            sj += 1
          }
          sj = 0
          si += chunkSize
        }
        arr
      }
      .flatMap { case ((i, j), ibses) =>
        val arr = new Array[((Int, Int), ExtendedIBDInfo)](chunkSize * chunkSize)
        var si = 0
        var sj = 0
        while (si != chunkSize) {
          while (sj != chunkSize) {
            arr(si * chunkSize + sj) =
              ((i * chunkSize + si, j * chunkSize + sj), ibses(si * chunkSize + sj))
            sj += 1
          }
          sj = 0
          si += 1
        }
        arr
      }
      .filter { case ((i, j), ibd) => j > i && j < nSamples && i < nSamples }
  }

  def generateComputeMaf(vaSignature: Type, computeMafExpr: String): (Variant, Annotation) => Double = {
    val mafSymbolTable = Map("v" -> (0, TVariant), "va" -> (1, vaSignature))
    val mafEc = EvalContext(mafSymbolTable)
    val computeMafThunk = Parser.parseTypedExpr[Double](computeMafExpr, mafEc)

    { (v: Variant, va: Annotation) =>
      mafEc.setAll(v, va)
      val maf = computeMafThunk()
        .getOrElse(fatal(s"The minor allele frequency expression evaluated to NA on variant $v."))

      if (maf < 0.0 || maf > 1.0)
        fatal(s"The minor allele frequency expression for $v evaluated to $maf which is not in [0,1].")

      maf
    }
  }

  def apply(vds: VariantDataset,
    computeMaf: Option[(Variant, Annotation) => Double] = None,
    bounded: Boolean = true,
    min: Option[Double] = None,
    max: Option[Double] = None): RDD[((String, String), ExtendedIBDInfo)] = {

    val sampleIds = vds.sampleIds

    computeIBDMatrix(vds, computeMaf, bounded)
      .filter { case (_, ibd) =>
        min.forall(ibd.ibd.PI_HAT >= _) &&
        max.forall(ibd.ibd.PI_HAT <= _) }
      .map { case ((i, j), ibd) => ((sampleIds(i), sampleIds(j)), ibd) }
  }
}

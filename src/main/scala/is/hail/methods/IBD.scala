package is.hail.methods

import is.hail.HailContext
import is.hail.expr.{EvalContext, Parser}
import is.hail.table.Table
import is.hail.annotations.{Annotation, Region, RegionValue, RegionValueBuilder, UnsafeRow}
import is.hail.expr.types._
import is.hail.variant.{GenomeReference, Genotype, HardCallView, MatrixTable, Variant}
import is.hail.methods.IBD.generateComputeMaf
import is.hail.rvd.RVD
import is.hail.stats.RegressionUtils
import org.apache.spark.rdd.RDD
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.language.higherKinds

object IBDInfo {
  def apply(Z0: Double, Z1: Double, Z2: Double): IBDInfo = {
    IBDInfo(Z0, Z1, Z2, Z1 / 2 + Z2)
  }

  val signature =
    TStruct(("Z0", TFloat64()), ("Z1", TFloat64()), ("Z2", TFloat64()), ("PI_HAT", TFloat64()))

  def fromRegionValue(rv: RegionValue): IBDInfo =
    fromRegionValue(rv.region, rv.offset)

  def fromRegionValue(region: Region, offset: Long): IBDInfo = {
    val Z0 = region.loadDouble(signature.loadField(region, offset, 0))
    val Z1 = region.loadDouble(signature.loadField(region, offset, 1))
    val Z2 = region.loadDouble(signature.loadField(region, offset, 2))
    val PI_HAT = region.loadDouble(signature.loadField(region, offset, 3))
    IBDInfo(Z0, Z1, Z2, PI_HAT)
  }
}

case class IBDInfo(Z0: Double, Z1: Double, Z2: Double, PI_HAT: Double) {
  def pointwiseMinus(that: IBDInfo): IBDInfo =
    IBDInfo(Z0 - that.Z0, Z1 - that.Z1, Z2 - that.Z2, PI_HAT - that.PI_HAT)

  def hasNaNs: Boolean = Array(Z0, Z1, Z2, PI_HAT).exists(_.isNaN)

  def toAnnotation: Annotation = Annotation(Z0, Z1, Z2, PI_HAT)

  def toRegionValue(rvb: RegionValueBuilder) {
    rvb.addDouble(Z0)
    rvb.addDouble(Z1)
    rvb.addDouble(Z2)
    rvb.addDouble(PI_HAT)
  }
}

object ExtendedIBDInfo {
  val signature =
    TStruct(("ibd", IBDInfo.signature), ("ibs0", TInt64()), ("ibs1", TInt64()), ("ibs2", TInt64()))

  def fromRegionValue(rv: RegionValue): ExtendedIBDInfo =
    fromRegionValue(rv.region, rv.offset)

  def fromRegionValue(region: Region, offset: Long): ExtendedIBDInfo = {
    val ibd = IBDInfo.fromRegionValue(region, signature.loadField(region, offset, 0))
    val ibs0 = region.loadLong(signature.loadField(region, offset, 1))
    val ibs1 = region.loadLong(signature.loadField(region, offset, 2))
    val ibs2 = region.loadLong(signature.loadField(region, offset, 3))
    ExtendedIBDInfo(ibd, ibs0, ibs1, ibs2)
  }
}

case class ExtendedIBDInfo(ibd: IBDInfo, ibs0: Long, ibs1: Long, ibs2: Long) {
  def pointwiseMinus(that: ExtendedIBDInfo): ExtendedIBDInfo =
    ExtendedIBDInfo(ibd.pointwiseMinus(that.ibd), ibs0 - that.ibs0, ibs1 - that.ibs1, ibs2 - that.ibs2)

  def hasNaNs: Boolean = ibd.hasNaNs

  def toAnnotation: Annotation = Annotation(ibd.toAnnotation, ibs0, ibs1, ibs2)

  def toRegionValue(rvb: RegionValueBuilder) {
    rvb.startStruct()
    ibd.toRegionValue(rvb)
    rvb.endStruct()
    rvb.addLong(ibs0)
    rvb.addLong(ibs1)
    rvb.addLong(ibs2)
  }
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

  def ibsForGenotypes(gs: HardCallView, maybeMaf: Option[Double]): IBSExpectations = {
    def calculateCountsFromMAF(maf: Double) = {
      var count = 0
      var i = 0
      while (i < gs.getLength) {
        gs.setGenotype(i)
        if (gs.hasGT) count += 1
        i += 1
      }
      val Na = count * 2.0
      val p = 1 - maf
      val q = maf
      val x = Na * p
      val y = Na * q
      (Na, x, y, p, q)
    }

    def estimateFrequenciesFromSample = {
      var na = 0
      var x = 0.0
      var i = 0
      while (i < gs.getLength) {
        gs.setGenotype(i)
        if (gs.hasGT) {
          na += 2
          x += countRefs(gs.getGT)
        }
        i += 1
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

  def computeIBDMatrix(vds: MatrixTable,
    computeMaf: Option[(RegionValue) => Double],
    min: Option[Double],
    max: Option[Double],
    sampleIds: IndexedSeq[String],
    bounded: Boolean): RDD[RegionValue] = {

    val nSamples = vds.nSamples

    val rowType = vds.rowType
    val unnormalizedIbse = vds.rdd2.mapPartitions { it =>
      val view = HardCallView(rowType)
      it.map { rv =>
        view.setRegion(rv)
        ibsForGenotypes(view, computeMaf.map(f => f(rv)))
      }
    }.fold(IBSExpectations.empty)(_ join _)

    val ibse = unnormalizedIbse.normalized

    val chunkedGenotypeMatrix = vds.rdd2.mapPartitions { it =>
      val view = HardCallView(rowType)
      it.map { rv =>
        view.setRegion(rv)
        Array.tabulate[Byte](view.getLength) { i =>
          view.setGenotype(i)
          if (view.hasGT)
            IBSFFI.gtToCRep(view.getGT)
          else
            IBSFFI.missingGTCRep
        }
      }
    }
      .zipWithIndex()
      .flatMap { case (gts, variantId) =>
        val vid = (variantId % chunkSize).toInt
        gts.grouped(chunkSize)
          .zipWithIndex
          .map { case (gtGroup, i) => ((i, variantId / chunkSize), (vid, gtGroup)) }
      }
      .aggregateByKey(Array.fill(chunkSize * chunkSize)(IBSFFI.missingGTCRep))({ case (x, (vid, gs)) =>
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
      .mapPartitions { it =>
        val region = Region()
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)
        for {
          ((iChunk, jChunk), ibses) <- it
          si <- (0 until chunkSize).iterator
          sj <- (0 until chunkSize).iterator
          i = iChunk * chunkSize + si
          j = jChunk * chunkSize + sj
          if j > i && j < nSamples && i < nSamples
          idx = si * chunkSize + sj
          eibd = calculateIBDInfo(ibses(idx * 3), ibses(idx * 3 + 1), ibses(idx * 3 + 2), ibse, bounded)
          if min.forall(eibd.ibd.PI_HAT >= _) && max.forall(eibd.ibd.PI_HAT <= _)
        } yield {
          region.clear()
          rvb.start(ibdSignature)
          rvb.startStruct()
          rvb.addString(sampleIds(i))
          rvb.addString(sampleIds(j))
          eibd.toRegionValue(rvb)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }
  }

  def apply(vds: MatrixTable,
    computeMafExpr: Option[String] = None,
    bounded: Boolean = true,
    min: Option[Double] = None,
    max: Option[Double] = None): Table = {

    min.foreach(min => optionCheckInRangeInclusive(0.0, 1.0)("minimum", min))
    max.foreach(max => optionCheckInRangeInclusive(0.0, 1.0)("maximum", max))
    vds.requireUniqueSamples("ibd")

    min.liftedZip(max).foreach { case (min, max) =>
      if (min > max) {
        fatal(s"minimum must be less than or equal to maximum: ${ min }, ${ max }")
      }
    }

    val computeMaf = computeMafExpr.map(generateComputeMaf(vds, _))
    val sampleIds = vds.sampleIds.asInstanceOf[IndexedSeq[String]]

    val ktRdd2 = computeIBDMatrix(vds, computeMaf, min, max, sampleIds, bounded)
    new Table(vds.hc, ktRdd2, ibdSignature, Array("i", "j"))
  }

  private val (ibdSignature, ibdMerger) = TStruct(("i", TString()), ("j", TString())).merge(ExtendedIBDInfo.signature)

  def toKeyTable(sc: HailContext, ibdMatrix: RDD[((Annotation, Annotation), ExtendedIBDInfo)]): Table = {
    val ktRdd = ibdMatrix.map { case ((i, j), eibd) => ibdMerger(Annotation(i, j), eibd.toAnnotation).asInstanceOf[Row] }
    Table(sc, ktRdd, ibdSignature, Array("i", "j"))
  }

  def toRDD(kt: Table): RDD[((Annotation, Annotation), ExtendedIBDInfo)] = {
    val rvd = kt.rvd
    rvd.map { rv =>
      val region = rv.region
      val i = TString.loadString(region, ibdSignature.loadField(rv, 0))
      val j = TString.loadString(region, ibdSignature.loadField(rv, 1))
      val ibd = IBDInfo.fromRegionValue(region, ibdSignature.loadField(rv, 2))
      val ibs0 = region.loadLong(ibdSignature.loadField(rv, 3))
      val ibs1 = region.loadLong(ibdSignature.loadField(rv, 4))
      val ibs2 = region.loadLong(ibdSignature.loadField(rv, 5))
      val eibd = ExtendedIBDInfo(ibd, ibs0, ibs1, ibs2)
      ((i, j), eibd)
    }
  }

  private[methods] def generateComputeMaf(vds: MatrixTable,
    computeMafExpr: String): (RegionValue) => Double = {

    val mafSymbolTable = Map("v" -> (0, vds.vSignature), "va" -> (1, vds.vaSignature))
    val mafEc = EvalContext(mafSymbolTable)
    val computeMafThunk = RegressionUtils.parseExprAsDouble(computeMafExpr, mafEc)
    val rowType = vds.rowType

    { (rv: RegionValue) =>
      val v = Variant.fromRegionValue(rv.region, rowType.loadField(rv, 1))
      val va = UnsafeRow.read(rowType.fieldTypes(2), rv.region, rowType.loadField(rv, 2))
      mafEc.setAll(v, va)
      val maf = computeMafThunk()

      if (maf == null)
        fatal(s"The minor allele frequency expression evaluated to NA on variant $v.")

      if (maf < 0.0 || maf > 1.0)
        fatal(s"The minor allele frequency expression for $v evaluated to $maf which is not in [0,1].")

      maf
    }
  }

}

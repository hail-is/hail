package is.hail.methods

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.sparkextras.ContextRDD
import is.hail.types.{MatrixType, TableType}
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PFloat64, PInt64}
import is.hail.types.virtual.{TFloat64, TStruct}
import is.hail.utils._
import is.hail.variant.{AllelePair, Call, Genotype, HardCallView}

import scala.language.higherKinds

import org.apache.spark.sql.Row

object IBDInfo {
  def apply(Z0: Double, Z1: Double, Z2: Double): IBDInfo =
    IBDInfo(Z0, Z1, Z2, Z1 / 2 + Z2)

  val pType =
    PCanonicalStruct(
      ("Z0", PFloat64()),
      ("Z1", PFloat64()),
      ("Z2", PFloat64()),
      ("PI_HAT", PFloat64()),
    )

  def fromRegionValue(offset: Long): IBDInfo = {
    val Z0 = Region.loadDouble(pType.loadField(offset, 0))
    val Z1 = Region.loadDouble(pType.loadField(offset, 1))
    val Z2 = Region.loadDouble(pType.loadField(offset, 2))
    val PI_HAT = Region.loadDouble(pType.loadField(offset, 3))
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
  val pType =
    PCanonicalStruct(
      ("ibd", IBDInfo.pType),
      ("ibs0", PInt64()),
      ("ibs1", PInt64()),
      ("ibs2", PInt64()),
    )

  def fromRegionValue(offset: Long): ExtendedIBDInfo = {
    val ibd = IBDInfo.fromRegionValue(pType.loadField(offset, 0))
    val ibs0 = Region.loadLong(pType.loadField(offset, 1))
    val ibs1 = Region.loadLong(pType.loadField(offset, 2))
    val ibs2 = Region.loadLong(pType.loadField(offset, 3))
    ExtendedIBDInfo(ibd, ibs0, ibs1, ibs2)
  }
}

case class ExtendedIBDInfo(ibd: IBDInfo, ibs0: Long, ibs1: Long, ibs2: Long) {
  def pointwiseMinus(that: ExtendedIBDInfo): ExtendedIBDInfo =
    ExtendedIBDInfo(
      ibd.pointwiseMinus(that.ibd),
      ibs0 - that.ibs0,
      ibs1 - that.ibs1,
      ibs2 - that.ibs2,
    )

  def hasNaNs: Boolean = ibd.hasNaNs

  def makeRow(i: Any, j: Any): Row = Row(i, j, ibd.toAnnotation, ibs0, ibs1, ibs2)

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
  E00: Double,
  E10: Double,
  E20: Double,
  E11: Double,
  E21: Double,
  E22: Double = 1,
  nonNaNCount: Int = 1,
) {
  def hasNaNs: Boolean = Array(E00, E10, E20, E11, E21).exists(_.isNaN)

  def normalized: IBSExpectations =
    IBSExpectations(
      E00 / nonNaNCount,
      E10 / nonNaNCount,
      E20 / nonNaNCount,
      E11 / nonNaNCount,
      E21 / nonNaNCount,
      E22,
      this.nonNaNCount,
    )

  def scaled(N: Long): IBSExpectations =
    IBSExpectations(E00 * N, E10 * N, E20 * N, E11 * N, E21 * N, E22 * N, this.nonNaNCount)

  def join(that: IBSExpectations): IBSExpectations =
    if (this.hasNaNs)
      that
    else if (that.hasNaNs)
      this
    else
      IBSExpectations(
        E00 + that.E00,
        E10 + that.E10,
        E20 + that.E20,
        E11 + that.E11,
        E21 + that.E21,
        nonNaNCount = nonNaNCount + that.nonNaNCount,
      )

}

object IBSExpectations {
  def empty: IBSExpectations = IBSExpectations(0, 0, 0, 0, 0, nonNaNCount = 0)
}

object IBD {
  def indicator(b: Boolean): Int = if (b) 1 else 0

  def countRefs(gtIdx: Int): Int = {
    val gt = Genotype.allelePair(gtIdx)
    indicator(AllelePair.j(gt) == 0) + indicator(AllelePair.k(gt) == 0)
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
          x += countRefs(Call.unphasedDiploidGtIndex(gs.getGT))
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

    val a00 =
      2 * p * p * q * q * ((x - 1) / x * (y - 1) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3)))
    val a10 =
      4 * p * p * p * q * ((x - 1) / x * (x - 2) / x * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3))) + 4 * p * q * q * q * ((y - 1) / y * (y - 2) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3)))
    val a20 =
      q * q * q * q * ((y - 1) / y * (y - 2) / y * (y - 3) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3))) + p * p * p * p * ((x - 1) / x * (x - 2) / x * (x - 3) / x * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3))) + 4 * p * p * q * q * ((x - 1) / x * (y - 1) / y * (Na / (Na - 1)) * (Na / (Na - 2)) * (Na / (Na - 3)))
    val a11 =
      2 * p * p * q * ((x - 1) / x * Na / (Na - 1) * Na / (Na - 2)) + 2 * p * q * q * ((y - 1) / y * Na / (Na - 1) * Na / (Na - 2))
    val a21 =
      p * p * p * ((x - 1) / x * (x - 2) / x * Na / (Na - 1) * Na / (Na - 2)) + q * q * q * ((y - 1) / y * (y - 2) / y * Na / (Na - 1) * Na / (Na - 2)) + p * p * q * ((x - 1) / x * Na / (Na - 1) * Na / (Na - 2)) + p * q * q * ((y - 1) / y * Na / (Na - 1) * Na / (Na - 2))
    IBSExpectations(a00, a10, a20, a11, a21)
  }

  def calculateIBDInfo(N0: Long, N1: Long, N2: Long, ibse: IBSExpectations, bounded: Boolean)
    : ExtendedIBDInfo = {
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

  def computeIBDMatrix(
    ctx: ExecuteContext,
    input: MatrixValue,
    computeMaf: Option[(RegionValue) => Double],
    min: Option[Double],
    max: Option[Double],
    sampleIds: IndexedSeq[String],
    bounded: Boolean,
  ): ContextRDD[Long] = {

    val nSamples = input.nCols
    val sm = ctx.stateManager

    val rowPType = input.rvRowPType
    val unnormalizedIbse = input.rvd.mapPartitions { (ctx, it) =>
      val rv = RegionValue(ctx.r)
      val view = HardCallView(rowPType)
      it.map { ptr =>
        rv.setOffset(ptr)
        view.set(ptr)
        ibsForGenotypes(view, computeMaf.map(f => f(rv)))
      }
    }.fold(IBSExpectations.empty)(_ join _)

    val ibse = unnormalizedIbse.normalized

    val chunkedGenotypeMatrix = input.rvd.mapPartitions { (_, it) =>
      val view = HardCallView(rowPType)
      it.map { ptr =>
        view.set(ptr)
        Array.tabulate[Byte](view.getLength) { i =>
          view.setGenotype(i)
          if (view.hasGT)
            IBSFFI.gtToCRep(Call.unphasedDiploidGtIndex(view.getGT))
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
      .aggregateByKey(Array.fill(chunkSize * chunkSize)(IBSFFI.missingGTCRep))(
        { case (x, (vid, gs)) =>
          for (i <- gs.indices) x(vid * chunkSize + i) = gs(i)
          x
        },
        { case (x, y) =>
          for (i <- y.indices)
            if (x(i) == IBSFFI.missingGTCRep)
              x(i) = y(i)
          x
        },
      )
      .map { case ((s, v), gs) => (v, (s, IBSFFI.pack(chunkSize, chunkSize, gs))) }

    val joined = ContextRDD.weaken(chunkedGenotypeMatrix.join(chunkedGenotypeMatrix)
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
      })

    joined
      .cmapPartitions { (ctx, it) =>
        val rvb = new RegionValueBuilder(sm, ctx.region)
        for {
          ((iChunk, jChunk), ibses) <- it
          si <- (0 until chunkSize).iterator
          sj <- (0 until chunkSize).iterator
          i = iChunk * chunkSize + si
          j = jChunk * chunkSize + sj
          if j > i && j < nSamples && i < nSamples
          idx = si * chunkSize + sj
          eibd =
            calculateIBDInfo(ibses(idx * 3), ibses(idx * 3 + 1), ibses(idx * 3 + 2), ibse, bounded)
          if min.forall(eibd.ibd.PI_HAT >= _) && max.forall(eibd.ibd.PI_HAT <= _)
        } yield {
          rvb.start(ibdPType)
          rvb.startStruct()
          rvb.addString(sampleIds(i))
          rvb.addString(sampleIds(j))
          eibd.toRegionValue(rvb)
          rvb.endStruct()
          rvb.end()
        }
      }
  }

  private val ibdPType =
    PCanonicalStruct(
      required = true,
      Array(
        ("i", PCanonicalString()),
        ("j", PCanonicalString()),
      ) ++ ExtendedIBDInfo.pType.fields.map(f => (f.name, f.typ)): _*
    )

  private val ibdKey = FastSeq("i", "j")

  private[methods] def generateComputeMaf(input: MatrixValue, fieldName: String)
    : (RegionValue) => Double = {
    val rvRowType = input.rvRowType
    val rvRowPType = input.rvRowPType
    val field = rvRowType.field(fieldName)
    assert(field.typ == TFloat64)
    val rowKeysF = input.typ.extractRowKey
    val entriesIdx = input.entriesIdx

    val idx = rvRowType.fieldIdx(fieldName)

    (rv: RegionValue) => {
      val isDefined = rvRowPType.isFieldDefined(rv.offset, idx)
      val maf = Region.loadDouble(rvRowPType.loadField(rv.offset, idx))
      if (!isDefined) {
        val row = new UnsafeRow(rvRowPType, rv).deleteField(entriesIdx)
        fatal(s"The minor allele frequency expression evaluated to NA at ${rowKeysF(row)}.")
      }
      if (maf < 0.0 || maf > 1.0) {
        val row = new UnsafeRow(rvRowPType, rv).deleteField(entriesIdx)
        fatal(
          s"The minor allele frequency expression for ${rowKeysF(row)} evaluated to $maf which is not in [0,1]."
        )
      }
      maf
    }
  }
}

case class IBD(
  mafFieldName: Option[String] = None,
  bounded: Boolean = true,
  min: Option[Double] = None,
  max: Option[Double] = None,
) extends MatrixToTableFunction {

  min.foreach(min => optionCheckInRangeInclusive(0.0, 1.0)("minimum", min))
  max.foreach(max => optionCheckInRangeInclusive(0.0, 1.0)("maximum", max))

  min.liftedZip(max).foreach { case (min, max) =>
    if (min > max) {
      fatal(s"minimum must be less than or equal to maximum: $min, $max")
    }
  }

  def preservesPartitionCounts: Boolean = false

  def typ(childType: MatrixType): TableType =
    TableType(IBD.ibdPType.virtualType, IBD.ibdKey, TStruct.empty)

  def execute(ctx: ExecuteContext, input: MatrixValue): TableValue = {
    input.requireUniqueSamples("ibd")
    val computeMaf = mafFieldName.map(IBD.generateComputeMaf(input, _))
    val crdd =
      IBD.computeIBDMatrix(ctx, input, computeMaf, min, max, input.stringSampleIds, bounded)
    TableValue(ctx, IBD.ibdPType, IBD.ibdKey, crdd)
  }
}

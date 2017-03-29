package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector}
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Genotype, VariantDataset}

object RegressionUtils {
  def toDouble(t: Type, code: String): Any => Double = t match {
    case TInt => _.asInstanceOf[Int].toDouble
    case TLong => _.asInstanceOf[Long].toDouble
    case TFloat => _.asInstanceOf[Float].toDouble
    case TDouble => _.asInstanceOf[Double]
    case TBoolean => _.asInstanceOf[Boolean].toDouble
    case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
  }

  def getPhenoCovCompleteSamples(
    vds: VariantDataset,
    ySA: String,
    covSA: Array[String]): (DenseVector[Double], DenseMatrix[Double], IndexedSeq[String]) = {

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val (yT, yQ) = Parser.parseExpr(ySA, ec)
    val yToDouble = toDouble(yT, ySA)
    val yIS = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      Option(yQ()).map(yToDouble)
    }

    val (covT, covQ0) = covSA.map(Parser.parseExpr(_, ec)).unzip
    val covQ = () => covQ0.map(_.apply())
    val covToDouble = (covT, covSA).zipped.map(toDouble)
    val covIS = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      (covQ().map(Option(_)), covToDouble).zipped.map(_.map(_))
    }

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, vds.sampleIds)
        .zipped
        .filter((y, c, s) => y.isDefined && c.forall(_.isDefined))

    val n = completeSamples.size
    if (n == 0)
      fatal("No complete samples: each sample is missing its phenotype or some covariate")

    val yArray = yForCompleteSamples.map(_.get).toArray
    if (yArray.toSet.size == 1)
      fatal(s"Constant phenotype: all complete samples have phenotype ${ yArray(0) }")
    val y = DenseVector(yArray)

    val k = covT.size
    val covArray = covForCompleteSamples.flatMap(1.0 +: _.map(_.get)).toArray
    val cov = new DenseMatrix(
      rows = n,
      cols = 1 + k,
      data = covArray,
      offset = 0,
      majorStride = 1 + k,
      isTranspose = true)

    (y, cov, completeSamples)
  }

  def setLastColumnToMaskedGts(X: DenseMatrix[Double], gts: IntIterator, mask: Array[Boolean]): Boolean = {
    require(X.offset == 0 && X.majorStride == X.rows && !X.isTranspose)

    val n = X.rows
    val k = X.cols - 1
    var missingIndices = new ArrayBuilder[Int]()
    var i = 0
    var j = k * n
    var gtSum = 0
    while (i < mask.length) {
      val gt = gts.nextInt()
      if (mask(i)) {
        if (gt != -1) {
          gtSum += gt
          X.data(j) = gt.toDouble
        } else
          missingIndices += j
        j += 1
      }
      i += 1
    }

    val missingIndicesArray = missingIndices.result()
    val nPresent = n - missingIndicesArray.length
    val gtMean = gtSum.toDouble / nPresent

    i = 0
    while (i < missingIndicesArray.length) {
      X.data(missingIndicesArray(i)) = gtMean
      i += 1
    }

    val lastColumnIsConstant = gtSum == 0 || gtSum == 2 * nPresent || (gtSum == nPresent && X.data.drop(n * k).forall(_ == 1d))

    !lastColumnIsConstant
  }

  // mean 0, norm sqrt(n), variance 1 (constant variants return None)
  def toNormalizedGtArray(gs: Iterable[Genotype], nSamples: Int): Option[Array[Double]] = {
    val gtVals = Array.ofDim[Double](nSamples)
    var nMissing = 0
    var gtSum = 0
    var gtSumSq = 0
    val gts = gs.hardCallIterator

    var i = 0
    while (i < nSamples) {
      val gt = gts.nextInt()
      gtVals(i) = gt
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          gtSum += 1
          gtSumSq += 1
        case 2 =>
          gtSum += 2
          gtSumSq += 4
        case -1 =>
          nMissing += 1
      }
      i += 1
    }

    val nPresent = nSamples - nMissing

    if (gtSum == 0 || gtSum == 2 * nPresent || gtSum == nPresent && gtSumSq == nPresent)
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      val gtMeanSqAll = (gtSumSq + nMissing * gtMean * gtMean) / nSamples
      val gtStdDevRec = 1d / math.sqrt(gtMeanSqAll - gtMean * gtMean)

      val gtDict = Array(0, (-gtMean) * gtStdDevRec, (1 - gtMean) * gtStdDevRec, (2 - gtMean) * gtStdDevRec)

      var j = 0
      while (j < nSamples) {
        gtVals(j) = gtDict(gtVals(j).toInt + 1)
        j += 1
      }

      Some(gtVals)
    }
  }

  // mean 0, norm approx. sqrt(m), variance approx. 1 (constant variants return None)
  def toHWENormalizedGtArray(gs: Iterable[Genotype], nSamples: Int, nVariants: Int): Option[Array[Double]] = {
    val gtVals = Array.ofDim[Double](nSamples)
    var nMissing = 0
    var gtSum = 0
    val gts = gs.hardCallIterator

    var i = 0
    while (i < nSamples) {
      val gt = gts.nextInt()
      gtVals(i) = gt
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          gtSum += 1
        case 2 =>
          gtSum += 2
        case -1 =>
          nMissing += 1
      }
      i += 1
    }

    val nPresent = nSamples - nMissing

    if (gtSum == 0 || gtSum == 2 * nPresent || gtSum == nPresent && gtVals.forall(gt => gt == 1 || gt == -1))
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      val p = 0.5 * gtMean
      val hweStdDevRec = 1d / math.sqrt(2 * p * (1 - p) * nVariants)

      val gtDict = Array(0, (-gtMean) * hweStdDevRec, (1 - gtMean) * hweStdDevRec, (2 - gtMean) * hweStdDevRec)

      var j = 0
      while (j < nSamples) {
        gtVals(j) = gtDict(gtVals(j).toInt + 1)
        j += 1
      }

      Some(gtVals)
    }
  }
}

// constructs SparseVector of genotype calls with missing values mean-imputed
// if all genotypes are missing then all elements are NaN
class SparseGtBuilder extends Serializable {
  private val missingRowIndices = new ArrayBuilder[Int]()
  private val rowsX = new ArrayBuilder[Int]()
  private val valsX = new ArrayBuilder[Double]()
  private var row = 0
  private var sparseLength = 0
  // current length of rowsX and valsX, used to track missingRowIndices
  private var nHet = 0
  private var nHomVar = 0

  def merge(gt: Int): SparseGtBuilder = {
    (gt: @unchecked) match {
      case 0 =>
      case 1 =>
        nHet += 1
        rowsX += row
        valsX += 1d
        sparseLength += 1
      case 2 =>
        nHomVar += 1
        rowsX += row
        valsX += 2d
        sparseLength += 1
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
    }
    row += 1

    this
  }

  def toSparseGtVectorAndStats(nSamples: Int): SparseGtVectorAndStats = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing
    val nHomRef = nSamples - nHet - nHomVar - nMissing
    val sumX = nHet + 2 * nHomVar
    val rowsXArray = rowsX.result()
    val valsXArray = valsX.result()

    val isConstant = nHomRef == nPresent || nHomVar == nPresent || nHet == nPresent || nPresent == 0

    val meanX = if (nPresent > 0) sumX.toDouble / nPresent else Double.NaN

    var i = 0
    while (i < missingRowIndicesArray.length) {
      valsXArray(i) = meanX
      i += 1
    }

    SparseGtVectorAndStats(new SparseVector(rowsXArray, valsXArray, nSamples), isConstant, meanX / 2, nHomRef, nHet, nHomVar, nMissing)
  }
}

case class SparseGtVectorAndStats(x: SparseVector[Double], isConstant: Boolean, af: Double, nHomRef: Int, nHet: Int, nHomVar: Int, nMissing: Int)

// constructs SparseVector of genotype calls (with missing values mean-imputed) in parallel with other statistics sufficient for linear regression
class LinRegBuilder(y: DenseVector[Double]) extends Serializable {
  private val missingRowIndices = new ArrayBuilder[Int]()
  private val rowsX = new ArrayBuilder[Int]()
  private val valsX = new ArrayBuilder[Double]()
  private var row = 0
  private var sparseLength = 0 // length of rowsX and valsX, used to track missingRowIndices
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private var sumYMissing = 0.0

  def merge(gt: Int): LinRegBuilder = {
    (gt: @unchecked) match {
      case 0 =>
      case 1 =>
        rowsX += row
        valsX += 1d
        sparseLength += 1
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case 2 =>
        rowsX += row
        valsX += 2d
        sparseLength += 1
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
        sumYMissing += y(row)
    }
    row += 1

    this
  }

  def stats(y: DenseVector[Double], n: Int, minAC: Int): Option[(SparseVector[Double], Double, Double)] = {
    require(minAC > 0)

    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = n - nMissing

    if (sumX < minAC || sumX == 2 * nPresent || sumX == nPresent && sumXX == nPresent)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      var i = 0
      while (i < missingRowIndicesArray.length) {
        valsXArray(missingRowIndicesArray(i)) = meanX
        i += 1
      }

      // variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      // assert(rowsXArray.isIncreasing)

      val x = new SparseVector[Double](rowsXArray, valsXArray, n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some((x, xx, xy))
    }
  }
}

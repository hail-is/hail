package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector}
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Genotype, VariantDataset}
import org.apache.spark.sql.Row

object RegressionUtils {
  def toDouble(t: Type, code: String): Any => Double = t match {
    case TInt => _.asInstanceOf[Int].toDouble
    case TLong => _.asInstanceOf[Long].toDouble
    case TFloat => _.asInstanceOf[Float].toDouble
    case TDouble => _.asInstanceOf[Double]
    case TBoolean => _.asInstanceOf[Boolean].toDouble
    case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
  }

  def getSampleAnnotation(vds: VariantDataset, annot: String, ec: EvalContext): IndexedSeq[Option[Double]] = {
    val (aT, aQ) = Parser.parseExpr(annot, ec)
    val aToDouble = toDouble(aT, annot)

    vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      Option(aQ()).map(aToDouble)
    }
  }

  // IndexedSeq indexed by samples, Array by annotations
  def getSampleAnnotations(vds: VariantDataset, annots: Array[String], ec: EvalContext): IndexedSeq[Array[Option[Double]]] = {
    val (aT, aQ0) = annots.map(Parser.parseExpr(_, ec)).unzip
    val aQ = () => aQ0.map(_.apply())
    val aToDouble = (aT, annots).zipped.map(toDouble)

    vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      (aQ().map(Option(_)), aToDouble).zipped.map(_.map(_))
    }
  }

  def getPhenoCovCompleteSamples(
    vds: VariantDataset,
    yExpr: String,
    covExpr: Array[String]): (DenseVector[Double], DenseMatrix[Double], IndexedSeq[Annotation]) = {

    val nCovs = covExpr.size + 1 // intercept

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotation(vds, yExpr, ec)
    val covIS = getSampleAnnotations(vds, covExpr, ec)

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

    val covArray = covForCompleteSamples.flatMap(1.0 +: _.map(_.get)).toArray
    val cov = new DenseMatrix(rows = n, cols = nCovs, data = covArray, offset = 0, majorStride = nCovs, isTranspose = true)

    if (n < vds.nSamples)
      warn(s"${vds.nSamples - n} of ${vds.nSamples} samples have a missing phenotype or covariate.")

    (y, cov, completeSamples)
  }

  def getPhenosCovCompleteSamples(
    vds: VariantDataset,
    yExpr: Array[String],
    covExpr: Array[String]): (DenseMatrix[Double], DenseMatrix[Double], IndexedSeq[Annotation]) = {

    val nPhenos = yExpr.size
    val nCovs = covExpr.size + 1 // intercept

    if (nPhenos == 0)
      fatal("No phenotypes present.")

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotations(vds, yExpr, ec)
    val covIS = getSampleAnnotations(vds, covExpr, ec)

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, vds.sampleIds)
        .zipped
        .filter((y, c, s) => y.forall(_.isDefined) && c.forall(_.isDefined))

    val n = completeSamples.size
    if (n == 0)
      fatal("No complete samples: each sample is missing its phenotype or some covariate")

    val yArray = yForCompleteSamples.flatMap(_.map(_.get)).toArray
    val y = new DenseMatrix(rows = n, cols = nPhenos, data = yArray, offset = 0, majorStride = nPhenos, isTranspose = true)

    val covArray = covForCompleteSamples.flatMap(1.0 +: _.map(_.get)).toArray
    val cov = new DenseMatrix(rows = n, cols = nCovs, data = covArray, offset = 0, majorStride = nCovs, isTranspose = true)

    if (n < vds.nSamples)
      warn(s"${vds.nSamples - n} of ${vds.nSamples} samples have a missing phenotype or covariate.")

    (y, cov, completeSamples)
  }

  def setLastColumnToMaskedGts(X: DenseMatrix[Double], gts: HailIterator[Double], mask: Array[Boolean], useHardCalls: Boolean): Boolean = {
    require(X.offset == 0 && X.majorStride == X.rows && !X.isTranspose)

    val nMaskedSamples = X.rows
    val k = X.cols - 1
    var missingIndices = new ArrayBuilder[Int]()
    var i = 0
    var j = k * nMaskedSamples
    var gtSum = 0d
    while (i < mask.length) {
      val gt = gts.next()
      if (mask(i)) {
        if (gt != -1) {
          gtSum += gt
          X.data(j) = gt
        } else
          missingIndices += j
        j += 1
      }
      i += 1
    }

    val missingIndicesArray = missingIndices.result()
    val nPresent = nMaskedSamples - missingIndicesArray.size

    if (nPresent > 0) {
      val gtMean = gtSum / nPresent

      i = 0
      while (i < missingIndicesArray.size) {
        X.data(missingIndicesArray(i)) = gtMean
        i += 1
      }
    }

    (useHardCalls &&
      !(gtSum == 0 || gtSum == 2 * nPresent || (gtSum == nPresent && X.data.drop(nMaskedSamples * k).forall(_ == 1d)))) ||
      (!useHardCalls && nPresent > 0)
  }

  //keyedRow consists of row key followed by numeric data (passed with key to avoid copying, key is ignored here)
  def setLastColumnToKeyedRow(X: DenseMatrix[Double], keyedRow: Row): Boolean = {
    val n = X.rows
    val k = X.cols - 1
    var missingRowIndices = new ArrayBuilder[Int]()
    var gtSum = 0d

    var i = 0
    var j = k * n
    while (i < n) {
      if (keyedRow.get(i + 1) == null)
        missingRowIndices += j + i
      else {
        val e = keyedRow.get(i + 1).asInstanceOf[java.lang.Number].doubleValue()
        X.data(j + i) = e
        gtSum += e
      }
      i += 1
    }

    val missingIndicesArray = missingRowIndices.result()
    val nMissing = missingIndicesArray.length
    val nPresent = n - nMissing
    val somePresent = nPresent > 0

    if (somePresent) {
      val gtMean = gtSum / nPresent

      i = 0
      while (i < missingIndicesArray.length) {
        X.data(missingIndicesArray(i)) = gtMean
        i += 1
      }
    }

    somePresent
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
      val gt = gts.next()
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
      val gtStdDevRec = 1 / math.sqrt(gtMeanSqAll - gtMean * gtMean)

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
      val gt = gts.next()
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
      val hweStdDevRec = 1 / math.sqrt(2 * p * (1 - p) * nVariants)

      val gtDict = Array(0, (-gtMean) * hweStdDevRec, (1 - gtMean) * hweStdDevRec, (2 - gtMean) * hweStdDevRec)

      var j = 0
      while (j < nSamples) {
        gtVals(j) = gtDict(gtVals(j).toInt + 1)
        j += 1
      }

      Some(gtVals)
    }
  }

  // constructs DenseVector of dosages (with missing values mean-imputed) in parallel with other statistics sufficient for linear regression
  def toLinregDosageStats(gs: Iterable[Genotype], y: DenseVector[Double], mask: Array[Boolean], minAC: Int): Option[(DenseVector[Double], Double, Double)] = {
    val nMaskedSamples = y.length
    val valsX = Array.ofDim[Double](nMaskedSamples)
    var sumX = 0d
    var sumXX = 0d
    var sumXY = 0d
    var sumYMissing = 0d
    var nMissing = 0
    val missingRowIndices = new ArrayBuilder[Int]()

    val gts = gs.dosageIterator
    var i = 0
    var row = 0
    while (gts.hasNext) {
      val gt = gts.next()
      if (mask(i)) {
        if (gt != -1) {
          valsX(row) = gt
          sumX += gt
          sumXX += gt * gt
          sumXY += gt * y(row)
        } else {
          nMissing += 1
          missingRowIndices += row
          sumYMissing += y(row)
        }
        row += 1
      }
      i += 1
    }

    val nPresent = nMaskedSamples - nMissing

    if (sumX < minAC)
      None
    else {
      val meanX = sumX / nPresent
      val missingRowIndicesArray = missingRowIndices.result()

      var i = 0
      while (i < missingRowIndicesArray.length) {
        valsX(missingRowIndicesArray(i)) = meanX
        i += 1
      }

      val x = DenseVector(valsX)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some(x, xx, xy)
    }
  }

  // constructs SparseVector of hard call genotypes (with missing values mean-imputed) in parallel with other statistics sufficient for linear regression
  def toLinregHardCallStats(gs: Iterable[Genotype], y: DenseVector[Double], mask: Array[Boolean], minAC: Int): Option[(SparseVector[Double], Double, Double)] = {
    val nMaskedSamples = y.length
    val lrb = new LinRegBuilder(y)
    val gts = gs.hardCallIterator

    var i = 0
    while (i < mask.length) {
      val gt = gts.next()
      if (mask(i))
        lrb.merge(gt)
      i += 1
    }

    lrb.stats(y, nMaskedSamples, minAC)
  }

  // constructs SparseVector of hard call genotypes (with missing values mean-imputed) in parallel with other summary statistics
  // if all genotypes are missing then all elements are NaN
  def toSparseHardCallStats(gs: Iterable[Genotype], mask: Array[Boolean], nMaskedSamples: Int): Option[(SparseVector[Double], Double)] = {
    val sb = new SparseGtBuilder()
    val gts = gs.hardCallIterator

    var i = 0
    while (i < mask.length) {
      val gt = gts.next()
      if (mask(i))
        sb.merge(gt)
      i += 1
    }

    sb.stats(nMaskedSamples)
  }

  //keyedRow consists of row key followed by numeric data (passed with key to avoid copying, key is ignored here)
  def statsKeyedRow(keyedRow: Row, y: DenseVector[Double]): Option[(DenseVector[Double], Double, Double)] = {
    val n = keyedRow.length - 1
    assert(y.length == n)

    val valsX = Array.ofDim[Double](n)
    var sumX = 0.0
    var sumXX = 0.0
    var sumXY = 0.0
    var sumYMissing = 0.0
    val missingRowIndices = new ArrayBuilder[Int]()

    var i = 0
    while (i < n) {
      if (keyedRow.get(i + 1) == null) {
        missingRowIndices += i
        sumYMissing += y(i)
      }
      else {
        val e = keyedRow.get(i + 1).asInstanceOf[java.lang.Number].doubleValue()
        valsX(i) = e
        sumX += e
        sumXX += e * e
        sumXY += e * y(i)
      }
      i += 1
    }

    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.length
    val nPresent = n - nMissing

    if (nPresent > 0) {
      val meanX = sumX / nPresent

      var j = 0
      while (j < nMissing) {
        valsX(missingRowIndicesArray(j)) = meanX
        j += 1
      }

      val x = DenseVector[Double](valsX)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some(x, xx, xy)
    }
    else
      None
  }
}

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
        valsX += 1
        sparseLength += 1
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case 2 =>
        rowsX += row
        valsX += 2
        sparseLength += 1
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0 // placeholder for meanX
        sparseLength += 1
        sumYMissing += y(row)
    }
    row += 1

    this
  }

  def stats(y: DenseVector[Double], nSamples: Int, minAC: Int): Option[(SparseVector[Double], Double, Double)] = {
    require(minAC > 0)

    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing

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

      val x = new SparseVector[Double](rowsXArray, valsXArray, nSamples)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some((x, xx, xy))
    }
  }
}

class SparseGtBuilder extends Serializable {
  private val missingRowIndices = new ArrayBuilder[Int]()
  private val rowsX = new ArrayBuilder[Int]()
  private val valsX = new ArrayBuilder[Double]()
  private var row = 0
  private var sparseLength = 0 // current length of rowsX and valsX, used to track missingRowIndices
  private var sumX = 0

  def merge(gt: Int): SparseGtBuilder = {
    (gt: @unchecked) match {
      case 0 =>
      case 1 =>
        rowsX += row
        valsX += 1
        sparseLength += 1
        sumX += 1
      case 2 =>
        rowsX += row
        valsX += 2
        sparseLength += 1
        sumX += 2
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0 // placeholder for meanX
        sparseLength += 1
    }
    row += 1

    this
  }

  def stats(nSamples: Int): Option[(SparseVector[Double], Double)] = {
    val rowsXArray = rowsX.result()
    val valsXArray = valsX.result()
    val missingRowIndicesArray = missingRowIndices.result()
    val nPresent = nSamples - missingRowIndicesArray.size

    if (sumX == 0 || sumX == 2 * nPresent || (sumX == nPresent && valsXArray.forall(_ == 1d)))
      None
    else {
      val meanX = if (nPresent > 0) sumX.toDouble / nPresent else Double.NaN

      var i = 0
      while (i < missingRowIndicesArray.length) {
        valsXArray(i) = meanX
        i += 1
      }

      val x = new SparseVector[Double](rowsXArray, valsXArray, nSamples)

      Some(x, meanX / 2)
    }
  }
}
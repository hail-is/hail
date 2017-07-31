package is.hail.stats

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.io.bgen.Bgen12GenotypeIterator
import is.hail.utils._
import is.hail.variant.{Genotype, VariantDataset}
import org.apache.spark.sql.Row

object RegressionUtils {
  // mask == null is interpreted as no mask
  // impute uses mean for missing gt rather than Double.NaN
  def hardCalls(gs: Iterable[Genotype], nKept: Int, mask: Array[Boolean] = null, impute: Boolean = true): SparseVector[Double] = {
    val gts = gs.hardCallIterator
    val rows = new ArrayBuilder[Int]()
    val vals = new ArrayBuilder[Double]()
    val missingSparseIndices = new ArrayBuilder[Int]()
    var i = 0
    var row = 0
    var sum = 0
    while (gts.hasNext) {
      val gt = gts.next()
      if (mask == null || mask(i)) {
        if (gt != 0) {
          rows += row
          if (gt != -1) {
            sum += gt
            vals += gt.toDouble
          } else {
            missingSparseIndices += vals.size
            vals += Double.NaN
          }
        }
        row += 1
      }
      i += 1
    }
    assert((mask == null || i == mask.length) && row == nKept)

    val valsArray = vals.result()
    val nMissing = missingSparseIndices.size
    if (impute) {
      val mean = sum.toDouble / (nKept - nMissing)
      i = 0
      while (i < nMissing) {
        valsArray(missingSparseIndices(i)) = mean
        i += 1
      }
    }

    new SparseVector[Double](rows.result(), valsArray, row)
  }

  def dosages(x: DenseVector[Double], gs: Iterable[Genotype], completeSampleIndex: Array[Int], missingSamples: ArrayBuilder[Int]) {
     genericDosages(x, gs, completeSampleIndex, missingSamples)
  }

  def dosages(gs: Iterable[Genotype], completeSampleIndex: Array[Int]): DenseVector[Double] = {
    val n = completeSampleIndex.length
    val x = new DenseVector[Double](n)
    val missingSamples = new ArrayBuilder[Int]()
    RegressionUtils.dosages(x, gs, completeSampleIndex, missingSamples)
    x
  }

  def genericDosages(x: DenseVector[Double], gs: Iterable[Genotype], completeSampleIndex: Array[Int], missingSamples: ArrayBuilder[Int]) {
    require(x.length == completeSampleIndex.length)

    missingSamples.clear()
    val n = completeSampleIndex.length
    val gts = gs.dosageIterator
    var i = 0
    var j = 0
    var sum = 0d
    while (j < n) {
      while (completeSampleIndex(j) > i) {
        gts.next()
        i += 1
      }
      assert(completeSampleIndex(j) == i)

      val gt = gts.next()
      i += 1
      if (gt != -1) {
        sum += gt
        x(j) = gt
      } else
        missingSamples += j
      j += 1
    }

    val nMissing = missingSamples.size
    val meanValue = sum / (n - nMissing)
    i = 0
    while (i < nMissing) {
      x(missingSamples(i)) = meanValue
      i += 1
    }
  }

  // keyedRow consists of row key followed by numeric data
  // impute uses mean for missing value rather than Double.NaN
  // if any non-missing value is Double.NaN, then mean is Double.NaN
  def keyedRowToVectorDouble(keyedRow: Row, impute: Boolean = true): DenseVector[Double] = {
    val n = keyedRow.length - 1
    val vals = Array.ofDim[Double](n)
    val missingRows = new ArrayBuilder[Int]()
    var sum = 0d
    var row = 0
    while (row < n) {
      val e0 = keyedRow.get(row + 1)
      if (e0 != null) {
        val e = e0.asInstanceOf[java.lang.Number].doubleValue()
        vals(row) = e
        sum += e
      } else
        missingRows += row
      row += 1
    }

    val nMissing = missingRows.size
    val missingValue = if (impute) sum / (n - nMissing) else Double.NaN
    var i = 0
    while (i < nMissing) {
      vals(missingRows(i)) = missingValue
      i += 1
    }

    DenseVector[Double](vals)
  }

  // !useHWE: mean 0, norm exactly sqrt(n), variance 1
  // useHWE: mean 0, norm approximately sqrt(m), variance approx. m / n
  // missing gt are mean imputed, constant variants return None, only HWE uses nVariants
  def normalizedHardCalls(gs: Iterable[Genotype], nSamples: Int, useHWE: Boolean = false, nVariants: Int = -1): Option[Array[Double]] = {
    require(!(useHWE && nVariants == -1))
    val gts = gs.hardCallIterator
    val vals = Array.ofDim[Double](nSamples)
    var nMissing = 0
    var sum = 0
    var sumSq = 0

    var row = 0
    while (row < nSamples) {
      val gt = gts.next()
      vals(row) = gt
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          sum += 1
          sumSq += 1
        case 2 =>
          sum += 2
          sumSq += 4
        case -1 =>
          nMissing += 1
      }
      row += 1
    }

    val nPresent = nSamples - nMissing
    val nonConstant = !(sum == 0 || sum == 2 * nPresent || sum == nPresent && sumSq == nPresent)

    if (nonConstant) {
      val mean = sum.toDouble / nPresent
      val stdDev = math.sqrt(
        if (useHWE)
          mean * (2 - mean) * nVariants / 2
        else {
          val meanSq = (sumSq + nMissing * mean * mean) / nSamples
          meanSq - mean * mean
        })

      val gtDict = Array(0, -mean / stdDev, (1 - mean) / stdDev, (2 - mean) / stdDev)
      var i = 0
      while (i < nSamples) {
        vals(i) = gtDict(vals(i).toInt + 1)
        i += 1
      }

      Some(vals)
    } else
      None
  }

  def toDouble(t: Type, code: String): Any => Double = t match {
    case TInt32 => _.asInstanceOf[Int].toDouble
    case TInt64 => _.asInstanceOf[Long].toDouble
    case TFloat32 => _.asInstanceOf[Float].toDouble
    case TFloat64 => _.asInstanceOf[Double]
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
    covExpr: Array[String]): (DenseVector[Double], DenseMatrix[Double], Array[Int]) = {

    val nCovs = covExpr.length + 1 // intercept

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotation(vds, yExpr, ec)
    val covIS = getSampleAnnotations(vds, covExpr, ec)

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, 0 until vds.nSamples)
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
      warn(s"${ vds.nSamples - n } of ${ vds.nSamples } samples have a missing phenotype or covariate.")

    (y, cov, completeSamples.toArray)
  }

  def getPhenosCovCompleteSamples(
    vds: VariantDataset,
    yExpr: Array[String],
    covExpr: Array[String]): (DenseMatrix[Double], DenseMatrix[Double], Array[Int]) = {

    val nPhenos = yExpr.length
    val nCovs = covExpr.length + 1 // intercept

    if (nPhenos == 0)
      fatal("No phenotypes present.")

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotations(vds, yExpr, ec)
    val covIS = getSampleAnnotations(vds, covExpr, ec)

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, 0 until vds.nSamples)
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
      warn(s"${ vds.nSamples - n } of ${ vds.nSamples } samples have a missing phenotype or covariate.")

    (y, cov, completeSamples.toArray)
  }

  // Retrofitting for 0.1, will be removed at 2.0 when constant checking is dropped (unless otherwise useful)
  def constantVector(x: Vector[Double]): Boolean = {
    require(x.size > 0)
    var curr = x(0)
    var i = 1
    while (i < x.size) {
      val next = x(i)
      if (next != curr) return false
      curr = next
      i += 1
    }
    true
  }
}
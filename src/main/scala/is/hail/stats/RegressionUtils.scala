package is.hail.stats

import breeze.linalg._
import is.hail.annotations.{Annotation, RegionValue}
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Genotype, HardCallView, VariantSampleMatrix}
import org.apache.spark.sql.Row

object RegressionUtils {
  def inputVector(x: DenseVector[Double],
    globalAnnotation: Annotation, sampleIds: IndexedSeq[Annotation], sampleAnnotations: IndexedSeq[Annotation],
    row: (Annotation, (Annotation, Iterable[Annotation])),
    ec: EvalContext,
    xf: () => java.lang.Double,
    completeSampleIndex: Array[Int],
    missingSamples: ArrayBuilder[Int]) {
    require(x.length == completeSampleIndex.length)

    val (v, (va, gs)) = row

    ec.setAll(globalAnnotation, v, va)

    missingSamples.clear()
    val n = completeSampleIndex.length
    val git = gs.iterator
    var i = 0
    var j = 0
    var sum = 0d
    while (j < n) {
      while (i < completeSampleIndex(j)) {
        git.next()
        i += 1
      }
      assert(completeSampleIndex(j) == i)

      val g = git.next()
      ec.set(3, sampleIds(i))
      ec.set(4, sampleAnnotations(i))
      ec.set(5, g)
      val dosage = xf()
      if (dosage != null) {
        sum += dosage
        x(j) = dosage
      } else
        missingSamples += j
      i += 1
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
  def normalizedHardCalls(view: HardCallView, nSamples: Int, useHWE: Boolean = false, nVariants: Int = -1): Option[Array[Double]] = {
    require(!(useHWE && nVariants == -1))
    val vals = Array.ofDim[Double](nSamples)
    var nMissing = 0
    var sum = 0
    var sumSq = 0

    var row = 0
    while (row < nSamples) {
      view.setGenotype(row)
      if (view.hasGT) {
        val gt = view.getGT
        vals(row) = gt
        (gt: @unchecked) match {
          case 0 =>
          case 1 =>
            sum += 1
            sumSq += 1
          case 2 =>
            sum += 2
            sumSq += 4
        }
      } else {
        vals(row) = -1
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

  def parseExprAsDouble(expr: String, ec: EvalContext): () => java.lang.Double = {
    val (xt, xf0) = Parser.parseExpr(expr, ec)

    def castToDouble[T](f: (T) => Double): () => java.lang.Double = { () =>
      val a = xf0()
      if (a == null)
        null
      else
        f(a.asInstanceOf[T])
    }

    xt match {
      case TInt32 => castToDouble[Int](_.toDouble)
      case TInt64 => castToDouble[Long](_.toDouble)
      case TFloat32 => castToDouble[Float](_.toDouble)
      case TFloat64 => () => xf0().asInstanceOf[java.lang.Double]
      case TBoolean => castToDouble[Boolean](_.toDouble)
      case _ => fatal(s"x expression `$expr' must be numeric or Boolean, got $xt")
    }
  }

  def getSampleAnnotation[RPK, RK, T >: Null](vsm: VariantSampleMatrix[RPK, RK, T], annot: String, ec: EvalContext): IndexedSeq[Option[Double]] = {
    val aQ = parseExprAsDouble(annot, ec)

    vsm.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      val a = aQ()
      if (a != null)
        Some(a: Double)
      else
        None
    }
  }

  // IndexedSeq indexed by samples, Array by annotations
  def getSampleAnnotations[RPK, RK, T >: Null](vds: VariantSampleMatrix[RPK, RK, T], annots: Array[String], ec: EvalContext): IndexedSeq[Array[Option[Double]]] = {
    val aQs = annots.map(parseExprAsDouble(_, ec))

    vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      aQs.map { aQ =>
        val a = aQ()
        if (a != null)
          Some(a: Double)
        else
          None
      }
    }
  }

  def getPhenoCovCompleteSamples[RPK, RK, T >: Null](
    vsm: VariantSampleMatrix[RPK, RK, T],
    yExpr: String,
    covExpr: Array[String]): (DenseVector[Double], DenseMatrix[Double], Array[Int]) = {

    val nCovs = covExpr.size + 1 // intercept

    val symTab = Map(
      "s" -> (0, vsm.sSignature),
      "sa" -> (1, vsm.saSignature))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotation(vsm, yExpr, ec)
    val covIS = getSampleAnnotations(vsm, covExpr, ec)

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, 0 until vsm.nSamples)
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

    if (n < vsm.nSamples)
      warn(s"${ vsm.nSamples - n } of ${ vsm.nSamples } samples have a missing phenotype or covariate.")

    (y, cov, completeSamples.toArray)
  }

  def getPhenosCovCompleteSamples[RPK, RK, T >: Null](
    vsm: VariantSampleMatrix[RPK, RK, T],
    yExpr: Array[String],
    covExpr: Array[String]): (DenseMatrix[Double], DenseMatrix[Double], Array[Int]) = {

    val nPhenos = yExpr.size
    val nCovs = covExpr.size + 1 // intercept

    if (nPhenos == 0)
      fatal("No phenotypes present.")

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vsm.saSignature))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotations(vsm, yExpr, ec)
    val covIS = getSampleAnnotations(vsm, covExpr, ec)

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, 0 until vsm.nSamples)
        .zipped
        .filter((y, c, s) => y.forall(_.isDefined) && c.forall(_.isDefined))

    val n = completeSamples.size
    if (n == 0)
      fatal("No complete samples: each sample is missing its phenotype or some covariate")

    val yArray = yForCompleteSamples.flatMap(_.map(_.get)).toArray
    val y = new DenseMatrix(rows = n, cols = nPhenos, data = yArray, offset = 0, majorStride = nPhenos, isTranspose = true)

    val covArray = covForCompleteSamples.flatMap(1.0 +: _.map(_.get)).toArray
    val cov = new DenseMatrix(rows = n, cols = nCovs, data = covArray, offset = 0, majorStride = nCovs, isTranspose = true)

    if (n < vsm.nSamples)
      warn(s"${ vsm.nSamples - n } of ${ vsm.nSamples } samples have a missing phenotype or covariate.")

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

  // Retrofitting for 0.1, will be removed at 2.0 when linreg AC is calculated post-imputation
  def hardCallsWithAC(gs: Iterable[Genotype], nKept: Int, mask: Array[Boolean] = null, impute: Boolean = true): (SparseVector[Double], Double) = {
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
    assert((mask == null || i == mask.size) && row == nKept)

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

    (new SparseVector[Double](rows.result(), valsArray, row), sum.toDouble)
  }
}

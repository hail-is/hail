package is.hail.rest

import breeze.linalg._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.json4s.jackson.Serialization.read

import scala.collection.mutable

case class GetRequestScoreCovariance(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  variant_list: Option[Array[SingleVariant]],
  compute_cov: Option[Boolean],
  limit: Option[Int],
  count: Option[Boolean]) extends GetRequest

case class GetResultScoreCovariance(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  active_variants: Option[Array[SingleVariant]],
  scores: Option[Array[Double]],
  covariance: Option[Array[Double]],
  sigma_sq: Option[Double],
  nsamples: Option[Int],
  count: Option[Int]) extends GetResult

class RestServiceScoreCovariance(vds: VariantDataset, covariates: Array[String], useDosages: Boolean, maxWidth: Int, hardLimit: Int) extends RestService { 
  private val nSamples: Int = vds.nSamples
  private val sampleMask: Array[Boolean] = Array.ofDim[Boolean](nSamples)
  private val availableCovariates: Set[String] = covariates.toSet
  private val availableCovariateToIndex: Map[String, Int] = covariates.zipWithIndex.toMap
  private val (sampleIndexToPresentness: Array[Array[Boolean]], 
                 covariateIndexToValues: Array[Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
  
  def readText(text: String): GetRequestScoreCovariance = read[GetRequestScoreCovariance](text)
  
  def getError(message: String, passback: Option[String]) = GetResultScoreCovariance(is_error = true, Some(message), passback, None, None, None, None, None, None)
  
  def getStats(req0: GetRequest): GetResultScoreCovariance = {
    val req = req0.asInstanceOf[GetRequestScoreCovariance]
    
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${ req.api_version }'. Supported API versions: 1")

    val yNameOpt = req.phenotype
    val (covNames, covVariants) = RestService.getCovariates(yNameOpt, req.covariates, availableCovariates)
    val (window, chrom, minPos, maxPos, minMAC, maxMAC) = RestService.getWindow(req.variant_filters, maxWidth)
    
    val variantsSet = mutable.Set[Variant]()

    req.variant_list.foreach { _.foreach { sv =>
      if (sv.chrom.isDefined && sv.pos.isDefined && sv.ref.isDefined && sv.alt.isDefined) {
        val v = Variant(sv.chrom.get, sv.pos.get, sv.ref.get, sv.alt.get)
        if (v.contig != chrom || v.start < minPos || v.start > maxPos)
          throw new RestFailure(s"Variant ${ v.toString } from 'variant_list' is not in the window ${ RestService.windowToString(window) }")
        variantsSet += v
      } else
        throw new RestFailure(s"All variants in 'variant_list' must include " +
          s"'chrom', 'pos', 'ref', and 'alt' fields: got " +
          s"${ (sv.chrom.getOrElse("NA"), sv.pos.getOrElse("NA"), sv.ref.getOrElse("NA"), sv.alt.getOrElse("NA")) }")
      }
    }

    // pull up?
    import is.hail.distributedmatrix.DistributedMatrix.implicits._
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._
    
    // filter and compute
    val filteredVds =
      if (variantsSet.isEmpty)
        vds.filterIntervals(IntervalTree(Array(window)), keep = true)
      else
        vds.filterVariantsList(variantsSet.toSet, keep = true)

    if (req.count.getOrElse(false)) {
      val count = filteredVds.countVariants().toInt // upper bound as does not account for MAC filtering
      
      GetResultScoreCovariance(is_error = false, None, req.passback, None, None, None, None, None, Some(count))
    } else {
      val (yOpt, cov) = RestService.getYCovAndSetMask(sampleMask, vds, window, yNameOpt, covNames, covVariants,
        useDosages, availableCovariates, availableCovariateToIndex, sampleIndexToPresentness, covariateIndexToValues)

      val (yResOpt, sigmaSqOpt) =
        if (yOpt.isDefined) {
          val y = yOpt.get
          val d = cov.rows - cov.cols
          if (d < 1)
            fatal(s"${cov.rows} samples and ${cov.cols} ${ plural(cov.cols, "covariate") } including intercept implies $d degrees of freedom.")
          
          val qrDecomp = qr.reduced(cov)
          val beta = qrDecomp.r \ (qrDecomp.q.t * y)
          val yRes = y - cov * beta
          val sigmaSq = (yRes dot yRes) / d
          
          (Some(yRes), Some(sigmaSq))
        } else
          (None, None)

      val n = cov.rows
      
      val (xOpt, activeVariants) = ToFilteredCenteredIndexedRowMatrix(filteredVds, n, sampleMask, minMAC, maxMAC)

      if (activeVariants.isEmpty) {
        GetResultScoreCovariance(is_error = false, None, req.passback,
          Some(Array.empty[SingleVariant]),
          Some(Array.empty[Double]),
          Some(Array.empty[Double]),
          sigmaSqOpt,
          Some(n),
          Some(0))
      } else {
        val limit = req.limit.getOrElse(hardLimit)
        
        if (activeVariants.length > limit)
          throw new RestFailure(s"Number of active variants $activeVariants exceeds limit $limit")

        val X = xOpt.get
        X.rows.persist()
        
        // consider block matrix route
        val scoresOpt = yResOpt.map { yRes =>
          val yResMat = new org.apache.spark.mllib.linalg.DenseMatrix(yRes.length, 1, yRes.toArray, true)
          X.multiply(yResMat).toBlockMatrixDense().toLocalMatrix().toArray
        }
        
        val covarianceOpt =
          if (req.compute_cov.isEmpty || (req.compute_cov.isDefined && req.compute_cov.get)) {
            val Xb = X.toBlockMatrixDense()
            val covarianceSquare = (Xb * Xb.transpose).toLocalMatrix().toArray
            Some(RestService.lowerTriangle(covarianceSquare, activeVariants.length))
          } else
            None
        
        X.rows.unpersist()

        val activeSingleVariants = activeVariants.map(v =>
          SingleVariant(Some(v.contig), Some(v.start), Some(v.ref), Some(v.alt)))
        
        GetResultScoreCovariance(is_error = false, None, req.passback,
          Some(activeSingleVariants),
          scoresOpt,
          covarianceOpt,
          sigmaSqOpt,
          Some(n),
          Some(activeVariants.length))
      }
    }
  }
}

// n = mask.filter(_).length
object ToFilteredCenteredIndexedRowMatrix {  
  def apply(vds: VariantDataset, n: Int, mask: Array[Boolean], minMAC: Int, maxMAC: Int): (Option[IndexedRowMatrix], Array[Variant]) = {
    require(vds.wasSplit)
    
    val filterAC = RestService.makeFilterAC(n, minMAC, maxMAC)

    val variantVectors = vds.rdd.flatMap { case (v, (_, gs)) =>  // when all missing, gets filtered by ac
      val (x, ac) = RegressionUtils.hardCallsWithAC(gs, n, mask)
      if (filterAC(ac)) {
        val mu = sum(x) / n // inefficient
        Some(v, Vectors.dense((x - mu).toArray))
      } else
        None
    }

    // order is preserved from original OrderedRDD
    val activeVariants = variantVectors.map(_._1).collect()
    
    val irmOpt = 
      if (!activeVariants.isEmpty) {
        val reducedIndicesBc = vds.sparkContext.broadcast(activeVariants.index)    
        val indexedRows = variantVectors.map { case (v, x) => IndexedRow(reducedIndicesBc.value(v), x) }
        Some(new IndexedRowMatrix(indexedRows, nRows = activeVariants.length, nCols = n))
      } else
        None

    (irmOpt, activeVariants)
  }
}
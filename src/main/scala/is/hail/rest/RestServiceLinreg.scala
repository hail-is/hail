package is.hail.rest

import breeze.linalg._
import is.hail.stats.{LinearRegressionModel, RegressionUtils}
import is.hail.variant.{Variant, VariantDataset, _}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.json4s.jackson.Serialization.read

import scala.collection.mutable

case class GetRequestLinreg(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  limit: Option[Int],
  count: Option[Boolean],
  sort_by: Option[Array[String]]) extends GetRequest {
}

case class GetResultLinreg(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  stats: Option[Array[LinregStat]],
  nsamples: Option[Int],
  count: Option[Int]) extends GetResult

case class LinregStat(chrom: String,
  pos: Int,
  ref: String,
  alt: String,
  `p-value`: Option[Double])

object RestServiceLinreg {
  def linreg(vds: VariantDataset, y: DenseVector[Double], cov: DenseMatrix[Double],
    sampleMask: Array[Boolean], useDosages: Boolean, minMAC: Int, maxMAC: Int): RDD[LinregStat] = {
    
    require(vds.wasSplit)
    require(minMAC >= 0 && maxMAC >= minMAC)
    
    val completeSampleIndex = (0 until vds.nSamples).filter(sampleMask).toArray

    val n = y.size
    val k = cov.cols
    val d = n - k - 1
    
    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")
       
    val filterAC = RestService.makeFilterAC(n, minMAC, maxMAC)
    
    info(s"Running linear regression on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))

    vds.rdd.map { case (v, (_, gs)) =>
      val (x: Vector[Double], ac) =
        if (!useDosages) // replace by hardCalls in 0.2, with ac post-imputation
          RegressionUtils.hardCallsWithAC(gs, n, sampleMaskBc.value)
        else {
          val x0 = RegressionUtils.dosages(gs, completeSampleIndexBc.value)
          (x0, sum(x0))
        }

      val optPval =
        if (filterAC(ac)) {
          val pval = LinearRegressionModel.fit(x, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d).p
          if (!pval.isNaN)
            Some(pval)
          else
            None
        }
        else
          None

      LinregStat(v.contig, v.start, v.ref, v.alt, optPval)
    }
  }
}

class RestServiceLinreg(vds: VariantDataset, covariates: Array[String], useDosages: Boolean, maxWidth: Int, hardLimit: Int) extends RestService { 
  private val nSamples: Int = vds.nSamples
  private val sampleMask: Array[Boolean] = Array.ofDim[Boolean](nSamples)
  private val availableCovariates: Set[String] = covariates.toSet
  private val availableCovariateToIndex: Map[String, Int] = covariates.zipWithIndex.toMap
  private val (sampleIndexToPresentness: Array[Array[Boolean]], 
                 covariateIndexToValues: Array[Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
  
  def readText(text: String): GetRequestLinreg = read[GetRequestLinreg](text)

  def getError(message: String, passback: Option[String]) = GetResultLinreg(is_error = true, Some(message), passback, None, None, None)
    
  def getStats(req0: GetRequest): GetResultLinreg = {
    val req = req0.asInstanceOf[GetRequestLinreg]
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1")

    val yName = req.phenotype.getOrElse { throw new RestFailure("Missing required field: phenotype") }
    val (covNames, covVariants) = RestService.getCovariates(req.phenotype, req.covariates, availableCovariates)
    val (window, chrom, minPos, maxPos, minMAC, maxMAC) = RestService.getWindow(req.variant_filters, maxWidth)

    // filter and compute
    val windowedVds = vds.filterIntervals(IntervalTree(Array(window)), keep = true)
    val count = windowedVds.countVariants().toInt
    
    val limit = req.limit.getOrElse(hardLimit)
    if (limit < 0)
      throw new RestFailure(s"limit must be non-negative: got $limit")
    if (count > limit) {
      throw new RestFailure(s"count of $count exceeds limit of $limit")
    }
    
    if (req.count.getOrElse(false))
      GetResultLinreg(is_error = false, None, req.passback, None, None, Some(count))
    else {
      val (Some(y), cov) = RestService.getYCovAndSetMask(sampleMask, vds, window, Some(yName), covNames, covVariants,
        useDosages, availableCovariates, availableCovariateToIndex, sampleIndexToPresentness, covariateIndexToValues)    
        
      var restStats = RestServiceLinreg.linreg(windowedVds, y, cov, sampleMask, useDosages, minMAC, maxMAC).collect()
      
      req.sort_by.foreach { sortFields => 
        if (!sortFields.areDistinct())
          throw new RestFailure("sort_by arguments must be distinct")
        
        val nRedundant =
          if (sortFields.endsWith(Array("pos", "ref", "alt")))
            3
          else if (sortFields.endsWith(Array("pos", "ref")))
            2
          else if (sortFields.endsWith(Array("pos")))
            1
          else
            0
        
        sortFields.dropRight(nRedundant).reverse.foreach { f =>
          restStats = f match {
            case "pos" => restStats.sortBy(_.pos)
            case "ref" => restStats.sortBy(_.ref)
            case "alt" => restStats.sortBy(_.alt)
            case "p-value" => restStats.sortBy(_.`p-value`.getOrElse(2d)) // missing values at end
            case _ => throw new RestFailure(s"Valid sort_by arguments are `pos', `ref', `alt', and `p-value': got $f")
          }
        }
      }
      GetResultLinreg(is_error = false, None, req.passback, Some(restStats), Some(y.size), Some(restStats.length))
    }
  }
}

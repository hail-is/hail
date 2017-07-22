package is.hail.rest

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.http4s.HttpService
import org.http4s.MediaType.`application/json`
import org.http4s.dsl._
import org.http4s.headers.`Content-Type`
import org.http4s.server.Router
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write

import scala.concurrent.ExecutionContext

object RestService {
  def inRangeFunc(n: Int, minMAC: Int, maxMAC: Int): Double => Boolean = {
    val lowerMinAC = 1 max minMAC
    val lowerMaxAC = n min maxMAC
    val upperMinAC = 2 * n - lowerMaxAC
    val upperMaxAC = 2 * n - lowerMinAC
    
    ac => (ac >= lowerMinAC && ac <= lowerMaxAC) || (ac >= upperMinAC && ac <= upperMaxAC)
  }
  
  def windowToString(window: Interval[Locus]): String =
    s"${ window.start.contig }:${ window.start.position }-${ window.end.position - 1 }"

  def getYCovAndSetMask(sampleMask: Array[Boolean],
    vds: VariantDataset,
    window: Interval[Locus],
    yNameOpt: Option[String],
    covNames: Array[String],
    covVariants: Array[Variant] = Array.empty[Variant],
    useDosages: Boolean,
    availableCovariates: Set[String],
    availableCovariateToIndex: Map[String, Int],
    sampleIndexToPresentness: Array[Array[Boolean]], 
    covariateIndexToValues: Array[Array[Double]]): (Option[DenseVector[Double]], DenseMatrix[Double]) = {

    val nSamples = vds.nSamples
    
    // sample mask
    val yCovIndices = (yNameOpt.toArray ++ covNames).map(availableCovariateToIndex).sorted

    var nMaskedSamples = 0
    var sampleIndex = 0
    while (sampleIndex < nSamples) {
      val include = yCovIndices.forall(sampleIndexToPresentness(sampleIndex))
      sampleMask(sampleIndex) = include
      if (include) nMaskedSamples += 1
      sampleIndex += 1
    }

    var arrayIndex = 0
    
    // y
    val yOpt = yNameOpt.map { yName =>
      val yArray = Array.ofDim[Double](nMaskedSamples)
      val yData = covariateIndexToValues(availableCovariateToIndex(yName))
      sampleIndex = 0
      while (sampleIndex < nSamples) {
        if (sampleMask(sampleIndex)) {
          yArray(arrayIndex) = yData(sampleIndex)
          arrayIndex += 1
        }
        sampleIndex += 1
      }
      DenseVector(yArray)
    }

    // cov: set intercept, phenotype covariate, and variant covariate values
    val nCovs = 1 + covNames.length + covVariants.length
    val covArray = Array.ofDim[Double](nMaskedSamples * nCovs)

    // intercept
    arrayIndex = 0
    while (arrayIndex < nMaskedSamples) {
      covArray(arrayIndex) = 1
      arrayIndex += 1
    }

    // phenotype covariates
    covNames.foreach { covName =>
      val thisCovData = covariateIndexToValues(availableCovariateToIndex(covName))
      sampleIndex = 0
      while (sampleIndex < nSamples) {
        if (sampleMask(sampleIndex)) {
          covArray(arrayIndex) = thisCovData(sampleIndex)
          arrayIndex += 1
        }
        sampleIndex += 1
      }
    }

    val completeSampleIndex = (0 until vds.nSamples).filter(sampleMask).toArray
    
    // variant covariates
    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)

    val covVariantWithGenotypes = vds
      .filterVariantsList(covVariants.toSet, keep = true)
      .rdd
      .map { case (v, (va, gs)) => (v,
        if (!useDosages)
          RegressionUtils.hardCalls(gs, nMaskedSamples, sampleMaskBc.value).toArray
        else
          RegressionUtils.dosages(gs, completeSampleIndexBc.value).toArray)
      }
      .collect()

    if (covVariantWithGenotypes.length < covVariants.length) {
      val missingVariants = covVariants.toSet.diff(covVariantWithGenotypes.map(_._1).toSet)
      throw new RestFailure(s"VDS does not contain variant ${ plural(missingVariants.size, "covariate") } ${ missingVariants.mkString(", ") }")
    }

    if (!covVariants.map(_.locus).forall(window.contains)) {
      val outlierVariants = covVariants.filter(v => !window.contains(v.locus))
      warn(s"Window ${ windowToString(window) } does not contain variant ${ plural(outlierVariants.length, "covariate") } ${ outlierVariants.mkString(", ") }. This may increase latency.")
    }

    var variantIndex = 0
    while (variantIndex < covVariants.length) {
      val thisCovGenotypes = covVariantWithGenotypes(variantIndex)._2
      var maskedSampleIndex = 0
      while (maskedSampleIndex < nMaskedSamples) {
        covArray(arrayIndex) = thisCovGenotypes(maskedSampleIndex)
        arrayIndex += 1
        maskedSampleIndex += 1
      }
      variantIndex += 1
    }
    val cov = new DenseMatrix[Double](nMaskedSamples, nCovs, covArray)

    (yOpt, cov)
  }
  
  def lowerTriangle(a: Array[Double], n: Int): Array[Double] = {
    val nTri = n * (n + 1) / 2
    val b = Array.ofDim[Double](nTri)
    var i = 0
    var j = n
    var k = 0
    var l = 0
    while (j > 0) {
      k = n - j
      while (k < n) {
        b(l) = a(i + k)
        k += 1
        l += 1
      }
      i += n
      j -= 1
    }
    
    b
  }
}

abstract class GetRequest {
  def passback: Option[String]
}

abstract class GetResult

abstract class RestService {
  def readText(text: String): GetRequest
  
  def getStats(req: GetRequest): GetResult
  
  def getError(message: String, passback: Option[String]): GetResult

  def service(implicit executionContext: ExecutionContext = ExecutionContext.global): HttpService =
    Router("" -> rootService(executionContext))
  
  def rootService(implicit executionContext: ExecutionContext) = HttpService {
    case _ -> Root =>
      // The default route result is NotFound. Sometimes MethodNotAllowed is more appropriate.
      MethodNotAllowed()

    case req@POST -> Root / "getStats" =>
      // println("in getStats")

      req.decode[String] { text =>
        info("request: " + text)

        implicit val formats = Serialization.formats(NoTypeHints)

        var passback: Option[String] = None
        try {
          val getStatsReq = readText(text)
          passback = getStatsReq.passback
          val result = getStats(getStatsReq)
          Ok(write(result))
            .putHeaders(`Content-Type`(`application/json`))
        } catch {
          case e: Exception =>
            val result = getError(e.getMessage, passback)
            BadRequest(write(result))
              .putHeaders(`Content-Type`(`application/json`))
        }
      }
  }
}

class RestFailure(message: String) extends Exception(message) {
  info(s"RestFailure: $message")
}

case class VariantFilter(operand: String,
  operator: String,
  value: String,
  operand_type: String)

case class Covariate(`type`: String,
  name: Option[String],
  chrom: Option[String],
  pos: Option[Int],
  ref: Option[String],
  alt: Option[String])

case class SingleVariant(chrom: Option[String],
  pos: Option[Int],
  ref: Option[String],
  alt: Option[String])

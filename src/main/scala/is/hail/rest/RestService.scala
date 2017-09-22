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

import scala.collection.mutable
import scala.concurrent.ExecutionContext

object RestService {
  // construct (covNames, covVariants)
  def getCovariates(
    phenotypeOpt: Option[String],
    covariatesOpt: Option[Array[Covariate]],
    availableCovariates: Set[String]): (Array[String], Array[Variant]) = {

    val covNamesSet = mutable.Set[String]()
    val covVariantsSet = mutable.Set[Variant]()
    
    covariatesOpt.foreach { covariates =>
      for (c <- covariates)
        c.`type` match {
          case "phenotype" =>
            c.name match {
              case Some(name) =>
                if (availableCovariates(name))
                  if (covNamesSet(name))
                    throw new RestFailure(s"Covariate $name is included as a covariate more than once")
                  else
                    covNamesSet += name
                else
                  throw new RestFailure(s"$name is not a valid covariate name")
              case None =>
                throw new RestFailure("Covariate of type 'phenotype' must include 'name' field in request")
            }
          case "variant" =>
            (c.chrom, c.pos, c.ref, c.alt) match {
              case (Some(chr), Some(pos), Some(ref), Some(alt)) =>
                val v = Variant(chr, pos, ref, alt)
                if (covVariantsSet(v))
                  throw new RestFailure(s"$v is included as a covariate more than once")
                else
                  covVariantsSet += v
              case missingFields =>
                throw new RestFailure("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request")
            }
          case other =>
            throw new RestFailure(s"Covariate type must be 'phenotype' or 'variant': got $other")
        }
    }
        
    phenotypeOpt.foreach { phenotype =>
      if (covNamesSet(phenotype))
        throw new RestFailure(s"$phenotype appears as both response phenotype and covariate")
      if (!availableCovariates(phenotype))
        throw new RestFailure(s"$phenotype is not a valid phenotype name")
    }
    
    (covNamesSet.toArray, covVariantsSet.toArray)
  }
  
  // construct (window, chrom, minPos, maxPos, minMAC, maxMAC)
  def getWindow(variant_filters: Option[Array[VariantFilter]], maxWidth: Int): (Interval[Locus], String, Int, Int, Int, Int) = {
        // construct variant filters
    var chrom = ""

    var minPos = 1
    var maxPos = Int.MaxValue // 2,147,483,647 is greater than length of longest chromosome

    var minMAC = 1
    var maxMAC = Int.MaxValue

    val nonNegIntRegEx = """\d+""".r

    variant_filters.foreach(_.foreach { f =>
      f.operand match {
        case "chrom" =>
          if (!(f.operator == "eq" && f.operand_type == "string"))
            throw new RestFailure(s"chrom filter operator must be 'eq' and operand_type must be 'string': got '${ f.operator }' and '${ f.operand_type }'")
          else if (f.value.isEmpty)
            throw new RestFailure("chrom filter value cannot be the empty string")
          else if (chrom.isEmpty)
            chrom = f.value
          else if (chrom != f.value)
            throw new RestFailure(s"Got incompatible chrom filters: '$chrom' and '${ f.value }'")
        case "pos" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"pos filter operand_type must be 'integer': got '${ f.operand_type }'")
          else if (!nonNegIntRegEx.matches(f.value))
            throw new RestFailure(s"Value of position in variant_filter must be a valid non-negative integer: got '${ f.value }'")
          else {
            val pos = f.value.toInt
            f.operator match {
              case "gte" => minPos = minPos max pos
              case "gt" => minPos = minPos max (pos + 1)
              case "lte" => maxPos = maxPos min pos
              case "lt" => maxPos = maxPos min (pos - 1)
              case "eq" => minPos = minPos max pos; maxPos = maxPos min pos
              case other =>
                throw new RestFailure(s"pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq': got '$other'")
            }
          }
        case "mac" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"mac filter operand_type must be 'integer': got '${ f.operand_type }'")
          else if (!nonNegIntRegEx.matches(f.value))
            throw new RestFailure(s"mac filter value must be a valid non-negative integer: got '${ f.value }'")
          val mac = f.value.toInt
          f.operator match {
            case "gte" => minMAC = minMAC max mac
            case "gt" => minMAC = minMAC max (mac + 1)
            case "lte" => maxMAC = maxMAC min mac
            case "lt" => maxMAC = maxMAC min (mac - 1)
            case other =>
              throw new RestFailure(s"mac filter operator must be 'gte', 'gt', 'lte', 'lt': got '$other'")
          }
        case other => throw new RestFailure(s"Filter operand must be 'chrom' or 'pos': got '$other'")
      }
    })

    // construct window
    if (chrom.isEmpty)
      throw new RestFailure("No chromosome specified in variant_filter")

    val width = maxPos - minPos
    if (width > maxWidth)
      throw new RestFailure(s"Interval length cannot exceed $maxWidth: got $width")

    if (width < 0)
      throw new RestFailure(s"Window is empty: got start $minPos and end $maxPos")

    val window = Interval(Locus(chrom, minPos), Locus(chrom, maxPos + 1))

    info(s"Using window ${ RestService.windowToString(window) } of size ${ width + 1 }")
    
    (window, chrom, minPos, maxPos, minMAC, maxMAC)
  }
  
  def makeFilterAC(n: Int, minMAC: Int, maxMAC: Int): Double => Boolean = {
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

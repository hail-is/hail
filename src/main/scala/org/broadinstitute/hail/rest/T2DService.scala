package org.broadinstitute.hail.rest

import collection.mutable
import org.apache.spark.sql.DataFrame
import org.broadinstitute.hail.methods.LinearRegressionHcs
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.utils._
import breeze.linalg.{DenseMatrix, DenseVector}
import org.http4s.headers.`Content-Type`
import org.http4s._
import org.http4s.MediaType._
import org.http4s.dsl._
import org.http4s.server._

import scala.concurrent.ExecutionContext
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{read, write}

case class VariantFilter(operand: String,
  operator: String,
  value: String,
  operand_type: String) {

  def filterDf(df: DataFrame, blockWidth: Int): DataFrame = {
    operand match {
      case "chrom" =>
        df.filter(df("contig") === "chr" + value)
      case "pos" =>
        var v = 0
        try
          v = value.toInt // FIXME: should I be catching this? Otherwise, error message given 1.5 is just "For input string: \"1.5\"
        catch {
          case e: Exception => throw new RESTFailure(s"Value of position in variant_filter must be an integer: got $value")
        }
        val vblock = v / blockWidth
        operator match {
          case "eq" =>
            df.filter(df("block") === vblock)
              .filter(df("start") === v)
          case "gte" =>
            df.filter(df("block") >= vblock)
              .filter(df("start") >= v)
          case "gt" =>
            df.filter(df("block") >= vblock)
              .filter(df("start") > v)
          case "lte" =>
            df.filter(df("block") <= vblock)
              .filter(df("start") <= v)
          case "lt" =>
            df.filter(df("block") <= vblock)
              .filter(df("start") < v)
        }
    }
  }
}


case class Covariate(`type`: String,
  name: Option[String],
  chrom: Option[String],
  pos: Option[Int],
  ref: Option[String],
  alt: Option[String])

case class GetStatsRequest(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  limit: Option[Int],
  count: Option[Boolean],
  sort_by: Option[Array[String]])

case class Stat(chrom: String,
  pos: Int,
  ref: String,
  alt: String,
  `p-value`: Option[Double])

case class GetStatsResult(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  stats: Option[Array[Stat]],
  nsamples: Option[Int],
  count: Option[Int])

class RESTFailure(message: String) extends Exception(message)

class T2DService(hcs: HardCallSet, covMap: Map[String, IndexedSeq[Option[Double]]], defaultMinMAC: Int = 0, maxWidth: Int = 600000, hardLimit: Int = 100000) {

  def getStats(req: GetStatsRequest): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RESTFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RESTFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1")

    val pheno = req.phenotype.getOrElse("T2D")
    val phenoCovs = mutable.Set[String]()
    val variantCovs = new mutable.ArrayBuffer[Variant]()

    req.covariates.foreach { covariates =>
      for (c <- covariates)
        c.`type` match {
          case "phenotype" =>
            c.name match {
              case Some(name) =>
                if (covMap.keySet(name))
                  phenoCovs += name
                else
                  throw new RESTFailure(s"$name is not a valid covariate name")
              case None =>
                throw new RESTFailure("Covariate of type 'phenotype' must include 'name' field in request")
            }
          case "variant" =>
            (c.chrom, c.pos, c.ref, c.alt) match {
              case (Some(chrom), Some(pos), Some(ref), Some(alt)) =>
                variantCovs += Variant(chrom, pos, ref, alt)
              case missingFields =>
                throw new RESTFailure("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request")
            }
          case other =>
            throw new RESTFailure(s"Supported covariate types are phenotype and variant: got $other")
        }
    }

    if (phenoCovs(pheno))
      throw new RESTFailure(s"$pheno appears as both the response phenotype and a covariate phenotype")

    // FIXME: I'm not satisfied with next section

    val reqCovMap = covMap.filterKeys(c => phenoCovs(c) || c == pheno)

    val reqSampleFilter: Array[Boolean] = hcs.sampleIds.indices.map(si => reqCovMap.valuesIterator.forall(_(si).isDefined)).toArray

    val n = hcs.nSamples

    val reqSampleOriginalIndex: Array[Int] = (0 until n).filter(reqSampleFilter).toArray

    val n0 = reqSampleOriginalIndex.size
    if (n0 == 0)
      throw new RESTFailure("No samples in the intersection of given phenotype and covariates.")

    val reduceSampleIndex: Array[Int] = Array.ofDim[Int](n)
    (0 until n0).foreach(i => reduceSampleIndex(reqSampleOriginalIndex(i)) = i) // FIXME: Map is more natural than Array, but less efficient when using nearly all samples

    val y: DenseVector[Double] =
      covMap.get(pheno) match {
        case Some(a) => DenseVector(reqSampleOriginalIndex.flatMap(a(_)))
        case None => throw new RESTFailure(s"$pheno is not a valid phenotype name")
      }

    val nCov = phenoCovs.size + variantCovs.size
    val covArray = phenoCovs.toArray.flatMap(c => reqSampleOriginalIndex.flatMap(covMap(c)(_))) ++ variantCovs.toArray.flatMap(v => hcs.variantGts(v, n0, reqSampleFilter, reduceSampleIndex))
    val cov: Option[DenseMatrix[Double]] =
      if (nCov > 0)
        Some(new DenseMatrix[Double](n0, nCov, covArray))
      else
        None

    var minPos = 0
    var maxPos = Int.MaxValue // 2,147,483,647 is greater than length of longest chromosome

    var minMAC = 0
    var maxMAC = Int.MaxValue

    val chromFilters = mutable.Set[VariantFilter]()
    val posFilters = mutable.Set[VariantFilter]()
    val macFilters = mutable.Set[VariantFilter]()

    var isSingleVariant = false
    var useDefaultMinMAC = true

    req.variant_filters.foreach(_.foreach { f =>
      f.operand match {
        case "chrom" =>
          if (!(f.operator == "eq" && f.operand_type == "string"))
            throw new RESTFailure(s"chrom filter operator must be 'eq' and operand_type must be 'string': got '${f.operator}' and '${f.operand_type}'")
          chromFilters += f
        case "pos" =>
          if (f.operand_type != "integer")
            throw new RESTFailure(s"pos filter operand_type must be 'integer': got '${f.operand_type}'")
          f.operator match {
            case "gte" => minPos = minPos max f.value.toInt
            case "gt" => minPos = minPos max (f.value.toInt + 1)
            case "lte" => maxPos = maxPos min f.value.toInt
            case "lt" => maxPos = maxPos min (f.value.toInt - 1)
            case "eq" => isSingleVariant = true
            case other =>
              throw new RESTFailure(s"pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq': got '$other'")
          }
          posFilters += f
        case "mac" =>
          if (f.operand_type != "integer")
            throw new RESTFailure(s"mac filter operand_type must be 'integer': got '${f.operand_type}'")
          f.operator match {
            case "gte" => minMAC = minMAC max f.value.toInt
            case "gt" => minMAC = minMAC max (f.value.toInt + 1)
            case "lte" => maxMAC = maxMAC min f.value.toInt
            case "lt" => maxMAC = maxMAC min (f.value.toInt - 1)
            case other =>
              throw new RESTFailure(s"mac filter operator must be 'gte', 'gt', 'lte', 'lt': got '$other'")
          }
          useDefaultMinMAC = false
        case other => throw new RESTFailure(s"Filter operand must be 'chrom' or 'pos': got '$other'")
      }
    })

    if (chromFilters.isEmpty)
      throw new RESTFailure("No chromosome specified in variant_filter")

    val width =
      if (isSingleVariant)
        1
      else
        maxPos - minPos

    if (width > maxWidth)
      throw new RESTFailure(s"Interval length cannot exceed $maxWidth: got $width")

    var df = hcs.df
    val blockWidth = hcs.blockWidth

    chromFilters.foreach(f => df = f.filterDf(df, blockWidth))
    posFilters.foreach(f => df = f.filterDf(df, blockWidth))

    if (useDefaultMinMAC)
      minMAC = defaultMinMAC

    val statsRDD = LinearRegressionHcs(hcs.copy(df = df), y, cov, reqSampleFilter, reduceSampleIndex, minMAC, maxMAC)
      .rdd
      .map { case (v, olrs) => Stat(v.contig, v.start, v.ref, v.alt, olrs.map(_.p)) }

    var stats =
      if (req.limit.isEmpty)
        statsRDD.collect() // avoids first pass of take, modify if stats grows beyond memory capacity
      else {
        val limit = req.limit.get
        if (limit < 0)
          throw new RESTFailure(s"limit must be non-negative: got $limit")
        statsRDD.take(limit)
      }

    if (stats.size > hardLimit)
      stats = stats.take(hardLimit)

    if (req.sort_by.isEmpty)
      stats = stats.sortBy(s => (s.pos, s.ref, s.alt))
    else {
      val sortFields = req.sort_by.get
      if (! sortFields.areDistinct())
        throw new RESTFailure("sort_by arguments must be distinct")

      //      var fields = a.toList
      //
      //      // Default sort order is [pos, ref, alt] and sortBy is stable
      //      if (fields.nonEmpty && fields.head == "pos") {
      //        fields = fields.tail
      //        if (fields.nonEmpty && fields.head == "ref") {
      //          fields = fields.tail
      //          if (fields.nonEmpty && fields.head == "alt")
      //            fields = fields.tail
      //        }
      //      }

      sortFields.reverse.foreach { f =>
        stats = f match {
          case "pos" => stats.sortBy(_.pos)
          case "ref" => stats.sortBy(_.ref)
          case "alt" => stats.sortBy(_.alt)
          case "p-value" => stats.sortBy(_.`p-value`.getOrElse(2d))
          case _ => throw new RESTFailure(s"Valid sort_by arguments are `pos', `ref', `alt', and `p-value': got $f")
        }
      }
    }

    if (req.count.getOrElse(false))
      GetStatsResult(is_error = false, None, req.passback, None, Some(n0), Some(stats.size)) // FIXME: don't bother to compute when just returning count
    else
      GetStatsResult(is_error = false, None, req.passback, Some(stats), Some(n0), Some(stats.size))
  }

  def service(implicit executionContext: ExecutionContext = ExecutionContext.global): HttpService = Router(
    "" -> rootService)

  def rootService(implicit executionContext: ExecutionContext) = HttpService {
    case _ -> Root =>
      // The default route result is NotFound. Sometimes MethodNotAllowed is more appropriate.
      MethodNotAllowed()

    case req@POST -> Root / "getStats" =>
      println("in getStats")

      req.decode[String] { text =>
        info("request: " + text)

        implicit val formats = Serialization.formats(NoTypeHints)

        var passback: Option[String] = None
        try {
          val getStatsReq = read[GetStatsRequest](text)
          passback = getStatsReq.passback
          val result = getStats(getStatsReq)
          Ok(write(result))
            .putHeaders(`Content-Type`(`application/json`))
        } catch {
          case e: Exception =>
            val result = GetStatsResult(is_error = true, Some(e.getMessage), passback, None, None, None)
            BadRequest(write(result))
              .putHeaders(`Content-Type`(`application/json`))
        }
      }
  }
}
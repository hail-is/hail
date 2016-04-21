package org.broadinstitute.hail.rest

import collection.mutable
import org.apache.spark.sql.DataFrame
import org.broadinstitute.hail.methods.LinearRegression
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
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
        assert(operand_type == "string"
          && operator == "eq")
        df.filter(df("contig") === "chr" + value)
      case "pos" =>
        assert(operand_type == "integer")
        val v = value.toInt
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
  count: Option[Int])

class RESTFailure(message: String) extends Exception(message)

class T2DService(hcs: HardCallSet, hcs1Mb: HardCallSet, hcs10Mb: HardCallSet, covMap: Map[String, Array[Double]]) {

  def getStats(req: GetStatsRequest): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RESTFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1.")
    }

    if (req.api_version != 1)
      throw new RESTFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1.")

    val MaxWidthForHcs = 600000
    val MaxWidthForHcs1Mb = 10000000
    val HardLimit = 100000 // max is around 16k for T2D

    val limit = req.limit.getOrElse(HardLimit)

    val y: DenseVector[Double] = {
      val pheno = req.phenotype.getOrElse("T2D")
      covMap.get(pheno) match {
        case Some(a) => DenseVector(a)
        case None => throw new RESTFailure(s"$pheno is not a valid phenotype name")
      }
    }

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
                  throw new RESTFailure(s"${c.name} is not a valid covariate name")
              case None =>
                throw new RESTFailure("Covariate of type 'phenotype' must include 'name' field in request.")
            }
          case "variant" =>
            (c.chrom, c.pos, c.ref, c.alt) match {
              case (Some(chrom), Some(pos), Some(ref), Some(alt)) =>
                variantCovs += Variant(chrom, pos, ref, alt)
              case missingFields =>
                throw new RESTFailure("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request.")
            }
          case other =>
            throw new RESTFailure(s"$other is not a supported covariate type.")
        }
    }

    val nCov = phenoCovs.size + variantCovs.size
    val covArray = phenoCovs.toArray.flatMap(s => covMap(s)) ++ variantCovs.toArray.flatMap(hcs.variantGts)
    val cov: Option[DenseMatrix[Double]] =
      if (nCov > 0)
        Some(new DenseMatrix[Double](hcs.nSamples, nCov, covArray))
      else
        None

    var minPos = 0
    var maxPos = 1000000000

    val chromFilters = mutable.Set[VariantFilter]()
    val posFilters = mutable.Set[VariantFilter]()

    var isSingleVariant = false

    req.variant_filters.foreach(_.foreach { f =>
      f.operand match {
        case "chrom" =>
          chromFilters += f
        case "pos" =>
          posFilters += f
          f.operator match {
            case "gte" => minPos = minPos max f.value.toInt
            case "gt" => minPos = minPos max (f.value.toInt + 1)
            case "lte" => maxPos = maxPos min f.value.toInt
            case "le" => maxPos = maxPos min (f.value.toInt - 1)
            case "eq" => isSingleVariant = true
            case other =>
              throw new RESTFailure(s"'pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq': '$other' not supported.")
          }
        case other => throw new RESTFailure(s"Filter operant must be 'chrom' or 'pos': '$other' not supported.")
      }
    })

    if (chromFilters.isEmpty)
      chromFilters += VariantFilter("chrom", "eq", "1", "string")

    val width =
      if (isSingleVariant)
        1
      else
        maxPos - minPos

    assert(maxPos >= minPos)

    val hcsToUse =
      if (width <= MaxWidthForHcs)
        hcs
      else if (width <= MaxWidthForHcs1Mb)
        hcs1Mb
      else
        hcs10Mb

    var df = hcsToUse.df
    val blockWidth = hcsToUse.blockWidth

    chromFilters.foreach(f => df = f.filterDf(df, blockWidth))
    posFilters.foreach(f => df = f.filterDf(df, blockWidth))

    var stats = LinearRegression(hcsToUse.copy(df = df), y, cov)
      .rdd
      .map { case (v, olrs) => Stat(v.contig, v.start, v.ref, v.alt, olrs.map(_.p)) }
      .collect()

    // FIXME: test timing with .take(limit) to avoid copying below

    if (stats.length > limit)
      stats = stats.take(limit)

    // test that sort terms are distinct.
    // if seq ends in contig or in (contig, pos), ignore them
    // don't sort multiple times

    req.sort_by match {
      case Some(fields) =>
        fields.reverse.foreach { f =>
          stats = f match {
            case "contig" => stats.sortBy(_.chrom)
            case "pos" => stats.sortBy(_.pos)
            case "ref" => stats.sortBy(_.ref)
            case "alt" => stats.sortBy(_.alt)
            case "p-value" => stats.sortBy(_.`p-value`.getOrElse(2d))
            case _ => throw new RESTFailure(s"Valid sort_by arguments are `contig', `pos', `ref', `alt', and `p-value': got $f")
          }
        }
      case None =>
    }

    if (req.count.getOrElse(false))
      GetStatsResult(is_error = false, None, req.passback, None, Some(stats.length))
    else
      GetStatsResult(is_error = false, None, req.passback, Some(stats), None)
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
            val result = GetStatsResult(is_error = true, Some(e.getMessage), passback, None, None)
            BadRequest(write(result))
              .putHeaders(`Content-Type`(`application/json`))
        }
      }
  }
}

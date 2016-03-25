package org.broadinstitute.hail.rest

import collection.mutable
import org.apache.spark.sql.DataFrame
import org.broadinstitute.hail.methods.{CovariateData, LinearRegressionOnHcs}
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
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{read, write}

case class VariantFilter(operand: String,
  operator: String,
  value: String,
  operand_type: String) {
  
  def filterDf(df: DataFrame): DataFrame = {
    operand match {
      case "chrom" =>
        assert(operand_type == "string"
          && operator == "eq")
        df.filter(df("contig") === "chr" + value)
      case "pos" =>
        assert(operand_type == "integer")
        val v = value.toInt
        val vblock = v / 100000
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

class T2DService(hcs: HardCallSet, cov: CovariateData) {

  def getStats(req: GetStatsRequest): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RESTFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1.")
    }

    if (req.api_version != 1)
      throw new RESTFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1.")
    
    val y: DenseVector[Double] = req.phenotype match {
      case Some(pheno) =>
        cov.covName.indexOf(pheno) match {
          case -1 => throw new RESTFailure(s"$pheno is not a valid phenotype name")
          case i => cov.data.get(::, i)
        }
      case None => throw new RESTFailure(s"Missing phenotype")
    }

    // GRCh37, http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/data/
    val chromEnd: Map[String, Int] = Map(
       "1" -> 249250621,
       "2" -> 243199373,
       "3" -> 198022430,
       "4" -> 191154276,
       "5" -> 180915260,
       "6" -> 171115067,
       "7" -> 159138663,
       "8" -> 146364022,
       "9" -> 141213431,
      "10" -> 135534747,
      "11" -> 135006516,
      "12" -> 133851895,
      "13" -> 115169878,
      "14" -> 107349540,
      "15" -> 102531392,
      "16" ->  90354753,
      "17" ->  81195210,
      "18" ->  78077248,
      "19" ->  59128983,
      "20" ->  63025520,
      "21" ->  48129895,
      "22" ->  51304566,
       "X" -> 155270560,
       "Y" ->  59373566)

    val phenoCovs = mutable.Set[String]()

    val variantCovs = new mutable.ArrayBuffer[Variant]()

    req.covariates.foreach { covariates =>
      for (c <- covariates)
        c.`type` match {
          case "phenotype" =>
            c.name match {
              case Some(name) =>
                if (cov.covName.contains(name))
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

    var cov2 = cov.filterCovariates(phenoCovs.toSet)

    if (variantCovs.nonEmpty)
      cov2 = cov2.appendCovariates(hcs.variantCovData(variantCovs.toArray))

    val HARDLIMIT = 17000
    val MAXWIDTH = 2000000

    val limit = req.limit.map(_.min(HARDLIMIT)).getOrElse(HARDLIMIT)

    var minPos = 0
    var maxPos = 1000000000

    val chromFilters = mutable.Set[VariantFilter]()
    val posFilters = mutable.Set[VariantFilter]()

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
            case other =>
              throw new RESTFailure(s"'pos filter operator must be 'gte', 'gt', 'lte', or 'lt': '$other' not supported.")
          }
        case other => throw new RESTFailure(s"Filter operant must be 'chrom' or 'pos': '$other' not supported.")
      }
    })

    if (chromFilters.isEmpty)
      chromFilters += VariantFilter("chrom", "eq", "1", "string")

    val width = maxPos - minPos

    if (width > MAXWIDTH)
      posFilters += VariantFilter("pos", "lte", (minPos + MAXWIDTH).toString, "integer")

    var df = hcs.df
    chromFilters.foreach(f => df = f.filterDf(df))
    posFilters.foreach(f => df = f.filterDf(df))

    val statsRDD = LinearRegressionOnHcs(hcs.copy(df = df), y, cov2)
      .rdd
      .map { case (v, olrs) => Stat(v.contig, v.start, v.ref, v.alt, olrs.map(_.p)) }

    val stats: Array[Stat] =
      if (width <= 600000)
        statsRDD.collect()
      else
        statsRDD.take(limit)

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

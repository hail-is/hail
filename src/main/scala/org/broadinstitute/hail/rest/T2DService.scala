package org.broadinstitute.hail.rest

import org.apache.spark.sql.DataFrame
import org.broadinstitute.hail.methods.{CovariateData, LinearRegressionOnHcs}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import breeze.linalg.DenseVector

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

  def filter(df: DataFrame): DataFrame = {
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
  name: String)

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
        throw new RESTFailure(s"Unknown md_version `$md_version'.  Available md_versions: mdv1.")
    }

    if (req.api_version != 1)
      throw new RESTFailure(s"Unsupported API version `${req.api_version}'.  Supported API versions: 1.")

    val y: DenseVector[Double] = req.phenotype match {
      case Some(pheno) =>
        cov.covName.indexOf(pheno) match {
          case -1 => throw new RESTFailure(s"$pheno is not a valid phenotype name")
          case i => cov.data.get(::, i)
        }
      case None => throw new RESTFailure(s"Missing phenotype")
    }

    def getCovName(c: Covariate): String =
      c.`type` match {
        case "phenotype" =>
          if (cov.covName.contains(c.name))
            c.name
          else
            throw new RESTFailure(s"${c.name} is not a valid covariate name")
        case "variant" =>
          throw new RESTFailure("\'variant\' is not a supported covariate type (yet)")
        case other =>
          throw new RESTFailure(s"$other is not a supported covariate type")
      }

    val cov2: CovariateData = req.covariates match {
      case Some(covsToKeep) => cov.filterCovariates(covsToKeep.map(getCovName).toSet)
      case None => cov.filterCovariates(Set())
    }

    val hardLimit = 20000
    val limit = req.limit.map(_.min(hardLimit)).getOrElse(hardLimit)

    var df = hcs.df
    req.variant_filters.foreach(
      _.foreach { f =>
        df = f.filter(df)
      })

    val stats: Array[Stat] = LinearRegressionOnHcs(hcs.copy(df = df), y, cov2)
      .rdd
      .map { case (v, olrs) => Stat(v.contig, v.start, v.ref, v.alt, olrs.map(_.p)) }
      .take(limit)

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

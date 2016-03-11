package org.broadinstitute.hail.rest

import org.broadinstitute.hail.methods.{CovariateData, LinearRegressionOnHcs}
import org.broadinstitute.hail.variant._
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


  def filter(hcs: HardCallSet): HardCallSet = {
    operand match {
      case "chrom" =>
        assert(operand_type == "string"
          && operator == "eq")
        hcs.filterVariants(v => v.contig == value)
      case "pos" =>
        assert(operand_type == "integer")
        val pos = value.toInt
        operator match {
          case "eq" =>
            hcs.filterVariants(v => v.start == pos)
          case "gte" =>
            hcs.filterVariants(v => v.start >= pos)
          case "gt" =>
            hcs.filterVariants(v => v.start > pos)
          case "lte" =>
            hcs.filterVariants(v => v.start <= pos)
          case "lt" =>
            hcs.filterVariants(v => v.start < pos)
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

object T2DService {
  var task: Server = _
  
  def getStats(req: GetStatsRequest, hcs1: HardCallSet, cov1: CovariateData): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RESTFailure(s"Unknown md_version `$md_version'.  Available md_versions: mdv1.")
    }

    if (req.api_version != 1)
      throw new RESTFailure(s"Unsupported API version `${req.api_version}'.  Supported API versions: 1.")

    var hcs: HardCallSet = hcs1
    req.variant_filters.foreach(_.foreach { filter =>
      hcs = filter.filter(hcs)
    })

    val y: DenseVector[Double] = req.phenotype match {
      case Some(pheno) => cov1.data(::, cov1.covName.indexOf(pheno))
      case None => throw new RESTFailure(s"Missing phenotype")
    }

    def getCovName(c: Covariate): String =
      c.`type` match {
        case "phenotype" =>
          if (cov1.covName.contains(c.name))
            c.name
          else
            throw new RESTFailure(s"${c.name} is not a valid covariate name")
        case "variant" => throw new RESTFailure("\'variant\' is not a supported covariate type (yet)")
        case other => throw new RESTFailure(s"$other is not a supported covariate type")
      }

    val cov: CovariateData = req.covariates match {
      case Some(covsToKeep) => cov1.filterCovariates(covsToKeep.map(getCovName).toSet)
      case None => cov1.filterCovariates(Set())
    }

    val hardLimit = 10000
    val limit = req.limit.map(_.min(hardLimit)).getOrElse(hardLimit)

    val stats: Array[Stat] = LinearRegressionOnHcs(hcs, y, cov)
      .rdd
      .map { case (v, olrs) => Stat(v.contig, v.start, v.ref, v.alt, olrs.map(_.p)) }
      .take(limit)

    if (req.count.getOrElse(false))
      GetStatsResult(is_error = false, None, req.passback, None, Some(stats.length))
    else
      GetStatsResult(is_error = false, None, req.passback, Some(stats), None)
  }

  def service(hcs: HardCallSet, cov: CovariateData)(implicit executionContext: ExecutionContext = ExecutionContext.global): HttpService = Router(
    "" -> rootService(hcs, cov))

  def rootService(hcs: HardCallSet, cov: CovariateData)(implicit executionContext: ExecutionContext) = HttpService {
    case _ -> Root =>
      // The default route result is NotFound. Sometimes MethodNotAllowed is more appropriate.
      MethodNotAllowed()

    case req@POST -> Root / "getStats" =>
      println("in getStats")

      // FIXME error handling
      req.decode[String] { text =>
        println(text)

        implicit val formats = Serialization.formats(NoTypeHints)

        // implicit val formats = DefaultFormats // Brings in default date formats etc.
        // val getStatsReq = parse(text).extract[GetStatsRequest]
        var passback: Option[String] = None
        try {
          val getStatsReq = read[GetStatsRequest](text)
          passback = getStatsReq.passback
          val result = getStats(getStatsReq, hcs, cov)
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

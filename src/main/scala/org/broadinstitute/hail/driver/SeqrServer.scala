package org.broadinstitute.hail.driver

import com.datastax.driver.core.{DataType, Session}
import com.datastax.driver.core.querybuilder.QueryBuilder
import org.apache.solr.client.solrj.{SolrClient, SolrQuery}
import org.apache.solr.client.solrj.impl.{CloudSolrClient, HttpSolrClient}
import org.broadinstitute.hail.Utils._
import org.http4s.headers.`Content-Type`
import org.http4s._
import org.http4s.MediaType._
import org.http4s.dsl._
import org.http4s.server._
import org.http4s.server.blaze.BlazeBuilder

import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{read, write}
import org.kohsuke.args4j.{Option => Args4jOption}

case class SeqrRequest(page: Int,
  limit: Int,
  sortBy: Option[Array[String]],
  variantFilters: Option[Map[String, JValue]],
  genotypeFilters: Option[Array[Map[String, JValue]]])

case class SeqrResponse(is_error: Boolean,
  page: Option[Int],
  limit: Option[Int],
  error_message: Option[String],
  found: Option[Long],
  variants: Option[Array[Map[String, JValue]]])

class SeqrService(solr: SolrClient, cassSession: Session, cassKeyspace: String, cassTable: String) {
  def expandException(e: Throwable): String =
    s"${ e.getClass.getName }: ${ e.getMessage }\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).foreach(exception => expandException(exception))
    }"

  def process(req: SeqrRequest): SeqrResponse = {
    // println(req)

    val q = req.variantFilters
      .map(_.map { case (field, JObject(List(("eq", value)))) =>
        s"+$field:${ value.values }"

      case (field, JObject(List(("range", JArray(List(from, to)))))) =>
        s"+$field:[${ from.values } TO ${ to.values }]"

      case (field, JObject(List(("in", JArray(elems))))) =>
        s"+$field:(${
          elems
            .map(_.values.toString)
            .mkString(" OR ")
        })"
      }
        .mkString(" "))
      .getOrElse("*:*")

    // println("q: ", q)
    var query = new SolrQuery(q)

    query.addField("chrom")
    query.addField("pos")
    query.addField("ref")
    query.addField("alt")

    query.setStart((req.page - 1) * req.limit)
    query.setRows(req.limit)

    val solrResponse = solr.query(query)
    // println(solrResponse)

    val docs = solrResponse.getResults
    val found = docs.getNumFound

    val variants = docs.asScala
      .flatMap { doc =>
        val chrom = doc.getFieldValue("chrom").asInstanceOf[String]
        val pos = doc.getFieldValue("pos").asInstanceOf[Int]
        val ref = doc.getFieldValue("ref").asInstanceOf[String]
        val alt = doc.getFieldValue("alt").asInstanceOf[String]

        val cassQuery = QueryBuilder.select()
          .all()
          .from(cassKeyspace, cassTable)
          .where()
          .and(QueryBuilder.eq("chrom", chrom))
          .and(QueryBuilder.eq("pos", pos))
          .and(QueryBuilder.eq("ref", ref))
          .and(QueryBuilder.eq("alt", alt))
        val cassResults = cassSession.execute(cassQuery)

        val cassColumns = cassResults.getColumnDefinitions
        cassResults.asScala.map { r =>
          cassColumns.asScala.zipWithIndex.flatMap { case (col, i) =>
            if (r.isNull(i))
              None
            else {
              val jv: JValue = (col.getType.getName: @unchecked) match {
                case DataType.Name.ASCII | DataType.Name.TEXT | DataType.Name.VARCHAR =>
                  JString(r.getString(i))
                case DataType.Name.INT =>
                  JInt(r.getInt(i))
                case DataType.Name.TINYINT =>
                  JInt(r.getByte(i).toInt)
                case DataType.Name.SMALLINT =>
                  JInt(r.getShort(i).toInt)
                case DataType.Name.BIGINT | DataType.Name.COUNTER =>
                  JInt(r.getLong(i))
                case DataType.Name.VARINT =>
                  JInt(r.getVarint(i))
                case DataType.Name.FLOAT =>
                  JDouble(r.getFloat(i))
                case DataType.Name.DOUBLE =>
                  JDouble(r.getDouble(i))

                case DataType.Name.LIST =>
                  val typeArgs = col.getType.getTypeArguments
                  assert(typeArgs.size() == 1)
                  (typeArgs.get(0).getName: @unchecked) match {
                    case DataType.Name.INT =>
                      JArray(r.getList(i, classOf[java.lang.Integer]).asScala
                          .map(v => JInt(v.toInt))
                          .toList)
                  }

                // FIXME: handle Set
              }
              Some((col.getName, jv))
            }
          }.toMap
        }
      }.toArray

    // println(variants.toSeq)

    SeqrResponse(is_error = false,
      Some(req.page),
      Some(req.limit),
      None,
      Some(found),
      Some(variants))
  }

  def service(implicit executionContext: ExecutionContext = ExecutionContext.global): HttpService = Router(
    "" -> rootService)

  def rootService(implicit executionContext: ExecutionContext) = HttpService {
    case req@POST -> Root =>
      req.decode[String] { text =>
        // println(text)

        implicit val formats = Serialization.formats(NoTypeHints)

        // implicit val formats = DefaultFormats // Brings in default date formats etc.
        // val getStatsReq = parse(text).extract[GetStatsRequest]
        var passback: Option[String] = None
        try {
          val req = read[SeqrRequest](text)
          val result = process(req)
          Ok(write(result))
            .putHeaders(`Content-Type`(`application/json`))
        } catch {
          case e: Exception =>
            println(expandException(e))
            val result = SeqrResponse(is_error = true, None, None, Some(e.getMessage), None, None)
            BadRequest(write(result))
              .putHeaders(`Content-Type`(`application/json`))
        }
      }
  }
}

object SeqrServerCommand extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-c",
      usage = "SolrCloud collection")
    var collection: String = _

    @Args4jOption(name = "-u", aliases = Array("--url"),
      usage = "Solr instance (URL) to connect to")
    var url: String = _

    @Args4jOption(name = "-z", aliases = Array("--zk-host"),
      usage = "Zookeeper host string to connect to")
    var zkHost: String = _

    @Args4jOption(required = true, name = "-a", aliases = Array("--address"),
      usage = "Cassandra contact point to connect to")
    var address: String = _

    @Args4jOption(required = true, name = "-k",
      usage = "Cassandra keyspace")
    var keyspace: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--table"),
      usage = "Cassandra table")
    var table: String = _

  }

  def newOptions = new Options

  def name = "seqrserver"

  def description = "Seqr backend server"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val vds = state.vds

    // FIXME unify with ExportVariantsSolr
    val url = options.url
    val zkHost = options.zkHost
    val collection = options.collection

    if (url == null && zkHost == null)
      fatal("one of -u or -z required")

    if (url != null && zkHost != null)
      fatal("both -u and -z given")

    if (zkHost != null && collection == null)
      fatal("-c required with -z")

    val solr =
      if (url != null)
        new HttpSolrClient(url)
      else {
        val cc = new CloudSolrClient(zkHost)
        cc.setDefaultCollection(collection)
        cc
      }

    val cassSession = CassandraStuff.getSession(options.address)

    val solrService = new SeqrService(solr, cassSession, options.keyspace, options.table)

    val task = BlazeBuilder.bindHttp(6060, "0.0.0.0")
      .mountService(solrService.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}

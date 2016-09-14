package org.broadinstitute.hail.driver

import com.datastax.driver.core.{DataType, Session}
import com.datastax.driver.core.querybuilder.QueryBuilder
import org.apache.solr.client.solrj.{SolrClient, SolrQuery}
import org.apache.solr.client.solrj.impl.{CloudSolrClient, HttpSolrClient}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.utils.StringEscapeUtils._
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
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.{read, write}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

case class SeqrRequest(page: Int,
  limit: Int,
  sort_by: Option[Array[String]],
  sample_filters: Option[Array[String]],
  variant_filters: Option[Map[String, JValue]],
  genotype_filters: Option[Map[String, Map[String, JValue]]])

case class SeqrResponse(is_error: Boolean,
  page: Option[Int],
  limit: Option[Int],
  error_message: Option[String],
  found: Option[Long],
  variants: Option[Array[Map[String, JValue]]])

class SeqrService(solrOnly: Boolean, solr: SolrClient, cassSession: Session, cassKeyspace: String, cassTable: String) {
  def unescapeString(s: String): String =
    unescapeStringSimple(s, '_')

  def expandException(e: Throwable): String =
    s"${ e.getClass.getName }: ${ e.getMessage }\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).foreach(exception => expandException(exception))
    }"

  def process(req: SeqrRequest): SeqrResponse = {
    // println(req)

    val filters =
      req.variant_filters.map(_.toArray).getOrElse(Array.empty) ++
        req.genotype_filters
          .map(_.toArray.flatMap { case (id, filt) =>
            filt.map { case (k, v) =>
              (id ++ "_" ++ k, v)
            }
          })
          .getOrElse(Array.empty)
    // println(filters.toSeq)

    val q = if (filters.isEmpty)
      "*:*"
    else {
      filters
        .map { case (field, JObject(List(("eq", value)))) =>
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
        .mkString(" ")
    }

    println("q: ", q)
    var query = new SolrQuery(q)

    if (!solrOnly) {
      query.addField("chrom")
      query.addField("start")
      query.addField("ref")
      query.addField("alt")
    }

    query.setStart((req.page - 1) * req.limit)
    query.setRows(req.limit)

    val solrResponse = solr.query(query)
    // println(solrResponse)

    val docs = solrResponse.getResults
    val found = docs.getNumFound

    val sampleFilters = req.sample_filters.map(_.toSet)

    val variants: Array[Map[String, JValue]] = if (solrOnly) {
      docs.asScala.map { doc =>
        val variants = mutable.Map.empty[String, JValue]
        val genotypes = mutable.Map.empty[String, mutable.Map[String, JValue]]

        doc.keySet.asScala.foreach { fname =>

          def toJSON(v: Any): JValue = v match {
            case null => JNull
            case b: java.lang.Boolean => JBool(b)
            case i: Int => JInt(i)
            case i: Long => JInt(i)
            case d: Double => JDouble(d)
            case f: Float => JDouble(f)
            case al: java.util.ArrayList[_] =>
              JArray(al.asScala.map(vi => toJSON(vi)).toList)
            case s: String => JString(s)
          }

          val jv = toJSON(doc.getFieldValue(fname))
          val sep = fname.indexOf("__")
          if (sep == -1)
            variants += ((fname, jv))
          else {
            val sample = fname.substring(0, sep)
            val include = sampleFilters.forall(_(sample))
            if (include) {
              val subFName = fname.substring(sep + 2)
              genotypes.updateValue(sample, mutable.Map.empty, { m =>
                m += ((subFName, jv))
                m
              })
            }
          }
        }

        variants += (("genotypes",
          JObject(genotypes.map { case (k, v) =>
            (k, JObject(v.toList))
          }.toList)))

        variants.toMap
      }.toArray
    } else {
      val prepared = cassSession.prepare(
        s"SELECT * FROM ${ cassKeyspace }.${ cassTable } WHERE chrom=? AND start=? AND ref=? AND alt=?")

      val futures = docs.asScala
        .map { doc =>
          val chrom = doc.getFieldValue("chrom")
          val start = doc.getFieldValue("start")
          val ref = doc.getFieldValue("ref")
          val alt = doc.getFieldValue("alt")

          cassSession.executeAsync(prepared.bind(chrom, start, ref, alt))
        }

      futures.flatMap { future =>
        val cassResults = future.getUninterruptibly()

        val cassColumns = cassResults.getColumnDefinitions
        cassResults.asScala.map { r =>
          cassColumns.asScala.zipWithIndex.flatMap { case (col, i) =>
            if (r.isNull(i))
              None
            else {
              val jv: JValue = (col.getType.getName: @unchecked) match {
                case DataType.Name.BOOLEAN =>
                  JBool(r.getBool(i))
                case DataType.Name.ASCII | DataType.Name.TEXT | DataType.Name.VARCHAR =>
                  JString(r.getString(i))
                case DataType.Name.TINYINT =>
                  JInt(r.getByte(i).toInt)
                case DataType.Name.SMALLINT =>
                  JInt(r.getShort(i).toInt)
                case DataType.Name.INT =>
                  JInt(r.getInt(i))
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
                    case DataType.Name.ASCII | DataType.Name.TEXT | DataType.Name.VARCHAR =>
                      JArray(r.getList(i, classOf[java.lang.String]).asScala
                        .map(v => JString(v))
                        .toList)
                    case DataType.Name.TINYINT | DataType.Name.SMALLINT |
                         DataType.Name.INT | DataType.Name.BIGINT |
                         DataType.Name.VARINT =>
                      JArray(r.getList(i, classOf[java.lang.Integer]).asScala
                        .map(v => JInt(v.toInt))
                        .toList)
                  }

                case DataType.Name.SET =>
                  val typeArgs = col.getType.getTypeArguments
                  assert(typeArgs.size() == 1)
                  (typeArgs.get(0).getName: @unchecked) match {
                    case DataType.Name.ASCII | DataType.Name.TEXT | DataType.Name.VARCHAR =>
                      JArray(r.getSet(i, classOf[java.lang.String]).asScala
                        .toList
                        .map(v => JString(v)))
                    case DataType.Name.TINYINT | DataType.Name.SMALLINT |
                         DataType.Name.INT | DataType.Name.BIGINT |
                         DataType.Name.VARINT =>
                      JArray(r.getSet(i, classOf[java.lang.Integer]).asScala
                        .toList
                        .map(v => JInt(v.toInt)))
                  }
              }
              Some((col.getName, jv))
            }
          }.toMap
        }
      }.toArray
    }

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
        try {
          // FIXME should be automatic
          val knownKeys = Set("page",
            "limit",
            "sort_by",
            "variant_filters",
            "genotype_filters")

          val json = parse(text)
          (json: @unchecked) match {
            case JObject(fields) =>
              fields.foreach { case (id, _) =>
                if (!knownKeys.contains(id))
                  throw new IllegalArgumentException(s"unknown field $id in request")
              }
          }

          val req = json.extract[SeqrRequest]
          // val req = read[SeqrRequest](text)

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

    @Args4jOption(name = "--solr-only", usage = "Return results directly queried from Solr")
    var solrOnly = false

    @Args4jOption(name = "-a", aliases = Array("--address"),
      usage = "Cassandra contact point to connect to")
    var address: String = _

    @Args4jOption(name = "-k", usage = "Cassandra keyspace")
    var keyspace: String = _

    @Args4jOption(name = "-t", aliases = Array("--table"),
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

    val solrOnly = options.solrOnly

    if ((url == null) == (zkHost == null))
      fatal("exactly one of -u or -z required")

    if (zkHost != null && collection == null)
      fatal("-c required with -z")

    if ((options.address == null) != (options.keyspace == null)
      || (options.address == null) != (options.table == null))
      fatal("none or all of -a, -k, -t required")

    if (solrOnly != (options.address == null))
      fatal("either --solr-only or all of -a, -k and -t required, but not both")

    val solr =
      if (url != null)
        new HttpSolrClient.Builder(url)
          .build()
      else {
        val cc = new CloudSolrClient.Builder()
          .withZkHost(zkHost)
          .build()
        cc.setDefaultCollection(collection)
        cc
      }

    val cassSession =
      if (solrOnly)
        null
      else
        CassandraStuff.getSession(options.address)

    val solrService = new SeqrService(solrOnly, solr, cassSession, options.keyspace, options.table)

    val task = BlazeBuilder.bindHttp(6060, "0.0.0.0")
      .mountService(solrService.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}

package is.hail.misc

import com.datastax.driver.core.{DataType, Session}
import is.hail.io.CassandraConnector
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import org.apache.solr.client.solrj.impl.{CloudSolrClient, HttpSolrClient}
import org.apache.solr.client.solrj.{SolrClient, SolrQuery}
import org.http4s.MediaType._
import org.http4s._
import org.http4s.dsl._
import org.http4s.headers.`Content-Type`
import org.http4s.server._
import org.http4s.server.blaze.BlazeBuilder
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.write
import org.json4s.jackson.{JsonMethods, Serialization}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.concurrent.ExecutionContext

case class SeqrRequest(page: Int,
  limit: Int,
  sort_by: Option[Array[String]],
  sample_ids: Option[Array[String]],
  variant_filters: Option[Map[String, JValue]],
  genotype_filters: Option[Map[String, Map[String, JValue]]])

case class SeqrResponse(is_error: Boolean,
  page: Option[Int],
  limit: Option[Int],
  error_message: Option[String],
  found: Option[Long],
  variants: Option[Array[Map[String, JValue]]])

class SeqrService(solrOnly: Boolean, jsonFields: Set[String], solr: SolrClient, cassSession: Session, cassKeyspace: String, cassTable: String) {
  def unescapeString(s: String): String =
    unescapeStringSimple(s, '_')

  def expandException(e: Throwable): String =
    s"${ e.getClass.getName }: ${ e.getMessage }\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).foreach(exception => expandException(exception))
    }"

  def unescapeColumnName(escaped: String): Either[String, (String, String)] = {
    val sep = escaped.indexOf("__")
    if (sep == -1)
      Left(unescapeString(escaped))
    else
      Right((unescapeString(escaped.substring(0, sep)),
        unescapeString(escaped.substring(sep + 2))))
  }

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

    // println("q: ", q)
    var query = new SolrQuery(q)

    if (!solrOnly) {
      query.addField("dataset_id")
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

    val sampleFilters = req.sample_ids.map(_.toSet)

    val variants: Array[Map[String, JValue]] = if (solrOnly) {
      docs.asScala.map { doc =>
        val variants = mutable.Map.empty[String, JValue]
        val genotypes = mutable.Map.empty[String, mutable.Map[String, JValue]]

        doc.keySet.asScala
          .filter(_ != "_version_")
          .foreach { name =>

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

            var jv = toJSON(doc.getFieldValue(name))
            unescapeColumnName(name) match {
              case Left(vfield) =>
                if (jsonFields.contains(vfield)) {
                  val JString(s) = jv
                  jv = JsonMethods.parse(s)
                }
                variants += ((vfield, jv))

              case Right((sample, gfield)) =>
                val include = sampleFilters.forall(_.contains(sample))
                if (include) {
                  genotypes.updateValue(sample, mutable.Map.empty, { m =>
                    m += ((gfield, jv))
                    m
                  })
                }
            }
          }

        if (genotypes.nonEmpty)
          variants += (("genotypes",
            JObject(genotypes.map { case (k, v) =>
              (k, JObject(v.toList))
            }.toList)))

        variants.toMap
      }.toArray
    } else {
      val prepared = cassSession.prepare(
        s"SELECT * FROM ${ cassKeyspace }.${ cassTable } WHERE dataset_id=? AND chrom=? AND start=? AND ref=? AND alt=?")

      val futures = docs.asScala
        .map { doc =>
          val dataset_id = doc.getFieldValue("dataset_id")
          val chrom = doc.getFieldValue("chrom")
          val start = doc.getFieldValue("start")
          val ref = doc.getFieldValue("ref")
          val alt = doc.getFieldValue("alt")

          cassSession.executeAsync(prepared.bind(dataset_id, chrom, start, ref, alt))
        }

      futures.map { future =>
        val cassResults = future.getUninterruptibly()

        val variants = mutable.Map.empty[String, JValue]
        val genotypes = mutable.Map.empty[String, mutable.Map[String, JValue]]

        val cassColumns = cassResults.getColumnDefinitions
        cassResults.asScala.foreach { r =>
          cassColumns.asScala.zipWithIndex.foreach { case (col, i) =>
            if (!r.isNull(i)) {
              var jv: JValue = (col.getType.getName: @unchecked) match {
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

              unescapeColumnName(col.getName) match {
                case Left(vfield) =>
                  if (jsonFields.contains(vfield)) {
                    val JString(s) = jv
                    jv = JsonMethods.parse(s)
                  }
                  variants += ((vfield, jv))

                case Right((sample, gfield)) =>
                  val include = sampleFilters.forall(_.contains(sample))
                  if (include) {
                    genotypes.updateValue(sample, mutable.Map.empty, { m =>
                      m += ((gfield, jv))
                      m
                    })
                  }
              }
            }
          }
        }

        if (genotypes.nonEmpty)
          variants += (("genotypes",
            JObject(genotypes.map { case (k, v) =>
              (k, JObject(v.toList))
            }.toList)))

        variants.toMap
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
            "sample_ids",
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

object SeqrServer {
  def start(collection: String = null,
    url: String = null,
    zkHost: String = null,
    jsonFields: String = null,
    solrOnly: Boolean = false,
    address: String = null,
    keyspace: String = null,
    table: String = null) {

    if ((url == null) == (zkHost == null))
      fatal("exactly one of -u or -z required")

    if (zkHost != null && collection == null)
      fatal("-c required with -z")

    if ((address == null) != (keyspace == null)
      || (address == null) != (table == null))
      fatal("none or all of -a, -k, -t required")

    if (solrOnly != (address == null))
      fatal("either --solr-only or all of -a, -k and -t required, but not both")

    val jsonFieldSet = jsonFields.split(",").map(_.trim).filter(_.nonEmpty).toSet


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
        CassandraConnector.getSession(address)

    val solrService = new SeqrService(solrOnly, jsonFieldSet, solr, cassSession, keyspace, table)

    val task = BlazeBuilder.bindHttp(6060, "0.0.0.0")
      .mountService(solrService.service, "/")
      .run
    task.awaitShutdown()
  }
}
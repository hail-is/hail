package is.hail.io

import java.util

import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.VariantSampleMatrix
import org.apache.solr.client.solrj.impl.{CloudSolrClient, HttpSolrClient}
import org.apache.solr.client.solrj.request.CollectionAdminRequest
import org.apache.solr.client.solrj.request.schema.SchemaRequest
import org.apache.solr.client.solrj.{SolrClient, SolrResponse}
import org.apache.solr.common.{SolrException, SolrInputDocument}
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.util.Random

object SolrConnector {

  def toSolrType(t: Type): String = t match {
    case _: TInt32 => "int"
    case _: TInt64 => "long"
    case _: TFloat32 => "float"
    case _: TFloat64 => "double"
    case _: TBoolean => "boolean"
    case _: TString => "string"
    // FIXME only 1 deep
    case i: TIterable => toSolrType(i.elementType)
    // FIXME
    case _ => fatal(s"invalid Solr type $t")
  }

  def addFieldReq(preexistingFields: Set[String], name: String, spec: Map[String, AnyRef], t: Type): Option[SchemaRequest.AddField] = {
    if (preexistingFields(name))
      return None

    var m = spec

    if (!m.contains("name"))
      m += "name" -> name

    if (!m.contains("type"))
      m += "type" -> toSolrType(t)

    if (!m.contains("stored"))
      m += "stored" -> true.asInstanceOf[AnyRef]

    if (!m.contains("multiValued"))
      m += "multiValued" -> t.isInstanceOf[TIterable].asInstanceOf[AnyRef]

    val req = new SchemaRequest.AddField(m.asJava)
    Some(req)
  }

  def documentAddField(document: SolrInputDocument, name: String, t: Type, value: Any) {
    if (t.isInstanceOf[TIterable]) {
      value.asInstanceOf[Traversable[_]].foreach { xi =>
        if (xi != null)
          document.addField(name, xi)
      }
    } else if (value != null)
      document.addField(name, value)
  }

  def processResponse(action: String, res: SolrResponse) {
    val tRes = res.getResponse.asScala.map { entry =>
      (entry.getKey, entry.getValue)
    }.toMap[String, AnyRef]

    tRes.get("errors") match {
      case Some(es) =>
        val errors = es.asInstanceOf[util.ArrayList[AnyRef]].asScala
        val error = errors.head.asInstanceOf[util.Map[String, AnyRef]]
          .asScala
        val errorMessages = error("errorMessages")
          .asInstanceOf[util.ArrayList[String]]
          .asScala
        fatal(s"error in $action:\n  ${ errorMessages.map(_.trim).mkString("\n    ") }${
          if (errors.length > 1)
            s"\n  and ${ errors.length - 1 } errors"
          else
            ""
        }")

      case None =>
        if (tRes.keySet != Set("responseHeader"))
          warn(s"unknown Solr response in $action: $res")
    }
  }

  def connect(zkHost: String, collection: String): SolrClient = {
    val cc = new CloudSolrClient.Builder()
      .withZkHost(zkHost)
      .build()
    cc.setDefaultCollection(collection)
    cc
  }

  def export(kt: KeyTable,
    zkHost: String,
    collection: String,
    blockSize: Int) {

    val sc = kt.hc.sc

    val solr = connect(zkHost, collection)

    // retrieve current fields
    val fieldsResponse = new SchemaRequest.Fields().process(solr)

    val preexistingFields = fieldsResponse.getFields.asScala
      .map(_.asScala("name").asInstanceOf[String])
      .toSet

    val addFieldReqs = kt.signature.fields.flatMap { f =>
      addFieldReq(preexistingFields, f.name, Map.empty, f.typ)
    }

    info(s"adding ${
      addFieldReqs.length
    } fields")

    if (addFieldReqs.nonEmpty) {
      val req = new SchemaRequest.MultiUpdate((addFieldReqs.toList: List[SchemaRequest.Update]).asJava)
      processResponse("add field request",
        req.process(solr))

      processResponse("commit",
        solr.commit())
    }

    solr.close()

    val localSignature = kt.signature
    val localBlockSize = blockSize
    val maxRetryInterval = 3 * 60 * 1000 // 3m

    kt.rdd.foreachPartition { it =>

      var solr: SolrClient = null

      it
        .grouped(localBlockSize)
        .foreach { block =>
          val documents = block.map { row =>
            val document = new SolrInputDocument()

            (row.asInstanceOf[Row].toSeq, localSignature.fields).zipped
              .foreach { case (a, f) =>
                documentAddField(document, f.name, f.typ, a)
              }

            document
          }

          var retry = true
          var retryInterval = 3 * 1000 // 3s

          while (retry) {
            try {
              if (solr == null)
                solr = connect(zkHost, collection)

              processResponse("add documents",
                solr.add(documents.asJava))
              retry = false
            } catch {
              case t: Throwable =>
                warn(s"caught exception while adding documents: ${
                  expandException(t, logMessage = true)
                }\n\tretrying")

                try {
                  solr.close()
                  solr = null
                } catch {
                  case t: Throwable =>
                    warn(s"caught exception while closing SorlClient: ${
                      expandException(t, logMessage = true)
                    }\n\tignoring")
                }

                Thread.sleep(Random.nextInt(retryInterval))
                retryInterval = (retryInterval * 2).max(maxRetryInterval)
            }
          }
        }

      var retry = true
      var retryInterval = 3 * 1000 // 3s

      while (retry) {
        try {
          if (solr == null)
            solr = connect(zkHost, collection)

          processResponse("commit",
            solr.commit())
          retry = false
        } catch {
          case t: Throwable =>
            warn(s"caught exception while committing: ${
              expandException(t, true)
            }\n\tretrying")

            try {
              solr.close()
              solr = null
            } catch {
              case t: Throwable =>
                warn(s"caught exception while closing SorlClient: ${
                  expandException(t, true)
                }\n\tignoring")
            }

            Thread.sleep(Random.nextInt(retryInterval))
            retryInterval = (retryInterval * 2).max(maxRetryInterval)
        }
      }

      solr.close()
      solr = null
    }
  }
}

package is.hail.io

import java.util

import is.hail.expr.{EvalContext, Parser, TBoolean, TDouble, TFloat, TGenotype, TInt, TIterable, TLong, TSample, TString, TVariant, Type}
import is.hail.utils.StringEscapeUtils.escapeStringSimple
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.solr.client.solrj.impl.{CloudSolrClient, HttpSolrClient}
import org.apache.solr.client.solrj.request.CollectionAdminRequest
import org.apache.solr.client.solrj.request.schema.SchemaRequest
import org.apache.solr.client.solrj.{SolrClient, SolrResponse}
import org.apache.solr.common.{SolrException, SolrInputDocument}

import scala.collection.JavaConverters._
import scala.util.Random

object SolrConnector {
  def toSolrType(t: Type): String = t match {
    case TInt => "int"
    case TLong => "long"
    case TFloat => "float"
    case TDouble => "double"
    case TBoolean => "boolean"
    case TString => "string"
    // FIXME only 1 deep
    case i: TIterable => toSolrType(i.elementType)
    // FIXME
    case _ => fatal("")
  }

  def escapeString(name: String): String =
    escapeStringSimple(name, '_', !_.isLetter, !_.isLetterOrDigit)

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

  def connect(url: String, zkHost: String, collection: String): SolrClient =
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

  def exportVariants(vds: VariantDataset,
    variantExpr: String,
    genotypeExpr: String,
    collection: String = null,
    url: String = null,
    zkHost: String = null,
    exportMissing: Boolean = false,
    exportRef: Boolean = false,
    drop: Boolean = false,
    numShards: Int = 1,
    blockSize: Int = 100) {

    val sc = vds.sparkContext
    val vas = vds.vaSignature
    val sas = vds.saSignature

    val vSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    val vparsed = Parser.parseSolrNamedArgs(variantExpr, vEC)

    val gSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas),
      "s" -> (2, TSample),
      "sa" -> (3, sas),
      "g" -> (4, TGenotype))
    val gEC = EvalContext(gSymTab)
    val gA = gEC.a

    val gparsed = Parser.parseSolrNamedArgs(genotypeExpr, gEC)

    if (url == null && zkHost == null)
      fatal("one of -u or -z required")

    if (url != null && zkHost != null)
      fatal("both -u and -z given")

    if (zkHost != null && collection == null)
      fatal("-c required with -z")

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

    //delete and re-create the collection
    if (drop) {
      try {
        CollectionAdminRequest.deleteCollection(collection).process(solr)
        info(s"deleted collection ${ collection }")
      } catch {
        case e: SolrException => warn(s"exportvariantssolr: unable to delete collection ${ collection }: ${ e }")
      }
    }

    try {
      //zookeeper needs to already have a pre-loaded config named "default_config".
      //if it doesn't, you can upload it using the solr command-line client:
      //   solr create_collection -c default_config; solr delete -c default_config
      //CollectionAdminRequest.listCollections().process(solr)
      CollectionAdminRequest.createCollection(collection, "default_config", numShards, 1).process(solr)
      info(s"created new solr collection ${ collection } with ${ numShards } shard" + (if (numShards > 1) "s" else ""))
    } catch {
      case e: SolrException => fatal(s"exportvariantssolr: unable to create collection ${ collection }: ${ e }")
    }

    // retrieve current fields
    val fieldsResponse = new SchemaRequest.Fields().process(solr)

    val preexistingFields = fieldsResponse.getFields.asScala
      .map(_.asScala("name").asInstanceOf[String])
      .toSet

    val addFieldReqs = vparsed.flatMap { case (name, spec, t, f) =>
      addFieldReq(preexistingFields, escapeString(name), spec, t)
    } ++ vds.sampleIds.flatMap { s =>
      gparsed.flatMap { case (name, spec, t, f) =>
        val fname = escapeString(s) + "__" + escapeString(name)
        addFieldReq(preexistingFields, fname, spec, t)
      }
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

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.sampleAnnotations)
    val localBlockSize = blockSize
    val maxRetryInterval = 3 * 60 * 1000 // 3m

    vds.rdd.foreachPartition { it =>

      var solr = connect(url, zkHost, collection)

      it
        .grouped(localBlockSize)
        .foreach { block =>

          val documents = block.map {
            case (v, (va, gs)) =>

              val document = new SolrInputDocument()

              vparsed.foreach {
                case (name, spec, t, f) =>
                  vEC.setAll(v, va)
                  f().foreach(x => documentAddField(document, escapeString(name), t, x))
              }

              gs.iterator.zipWithIndex.foreach {
                case (g, i) =>
                  if ((exportMissing || g.isCalled) && (exportRef || !g.isHomRef)) {
                    val s = sampleIdsBc.value(i)
                    val sa = sampleAnnotationsBc.value(i)
                    gparsed.foreach {
                      case (name, spec, t, f) =>
                        gEC.setAll(v, va, s, sa, g)
                        // __ can't appear in escaped string
                        f().foreach(x => documentAddField(document, escapeString(s) + "__" + escapeString(name), t, x))
                    }
                  }
              }

              document
          }

          var retry = true
          var retryInterval = 3 * 1000 // 3s

          while (retry) {
            try {
              processResponse("add documents",
                solr.add(documents.asJava))
              retry = false
            } catch {
              case t: Throwable =>
                warn(s"caught exception while adding documents: ${
                  expandException(t)
                }\n\tretrying")

                try {
                  solr.close()
                } catch {
                  case t: Throwable =>
                    warn(s"caught exception while closing SorlClient: ${
                      expandException(t)
                    }\n\tignoring")
                }

                solr = connect(url, zkHost, collection)

                Thread.sleep(Random.nextInt(retryInterval))
                retryInterval = (retryInterval * 2).max(maxRetryInterval)
            }
          }
        }

      processResponse("commit",
        solr.commit())

      solr.close()
    }
  }
}

package org.broadinstitute.hail.driver

import org.apache.solr.client.solrj.impl.{CloudSolrClient, HttpSolrClient}
import org.apache.solr.client.solrj.request.schema.SchemaRequest
import org.apache.solr.common.{SolrException, SolrInputDocument}
import org.broadinstitute.hail.expr._

import scala.collection.JavaConverters._
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.util.Random

object ExportVariantsSolr extends Command with Serializable {

  class Options extends BaseOptions {
    @Args4jOption(name = "-c",
      usage = "SolrCloud collection")
    var collection: String = _

    @Args4jOption(required = true, name = "-g",
      usage = "comma-separated list of fields/computations to be exported")
    var genotypeCondition: String = _

    @Args4jOption(name = "-u", aliases = Array("--url"),
      usage = "Solr instance (URL) to connect to")
    var url: String = _

    @Args4jOption(required = true, name = "-v",
      usage = "comma-separated list of fields/computations to be exported")
    var variantCondition: String = _

    @Args4jOption(name = "-z", aliases = Array("--zk-host"),
      usage = "Zookeeper host string to connect to")
    var zkHost: String = _

  }

  def newOptions = new Options

  def name = "exportvariantssolr"

  def description = "Export variant information to Solr"

  def supportsMultiallelic = true

  def requiresVDS = true

  def toSolrType(t: Type): String = t match {
    case TInt => "tint"
    case TLong => "tlong"
    case TFloat => "tfloat"
    case TDouble => "tdouble"
    case TBoolean => "boolean"
    case TString => "string"
    // FIXME only 1 deep
    case i: TIterable => toSolrType(i.elementType)
      // FIXME
    case _ => fatal("")
  }

  def escapeSolrFieldName(name: String): String = {
    val sb = new StringBuilder

    if (name.head.isDigit)
      sb += '_'

    name.foreach { c =>
      if (c.isLetterOrDigit)
        sb += c
      else
        sb += '_'
    }

    sb.result()
  }

  def addFieldReq(preexistingFields: Set[String], name: String, t: Type): Option[SchemaRequest.AddField] = {
    val escapedName = escapeSolrFieldName(name)
    if (preexistingFields(escapedName))
      return None

    val m = mutable.Map.empty[String, AnyRef]

    // FIXME check type

    m += "name" -> escapedName
    m += "type" -> toSolrType(t)
    m += "stored" -> true.asInstanceOf[AnyRef]
    if (t.isInstanceOf[TIterable])
      m += "multiValued" -> true.asInstanceOf[AnyRef]

    val req = new SchemaRequest.AddField(m.asJava)
    Some(req)
  }

  def documentAddField(document: SolrInputDocument, name: String, t: Type, value: Any) {
    if (t.isInstanceOf[TIterable]) {
      value.asInstanceOf[Seq[_]].foreach { xi =>
        document.addField(escapeSolrFieldName(name), xi)
      }
    } else
      document.addField(escapeSolrFieldName(name), value)
  }

  def run(state: State, options: Options): State = {
    val sc = state.vds.sparkContext
    val vds = state.vds
    val vas = vds.vaSignature
    val sas = vds.saSignature
    val gCond = options.genotypeCondition
    val vCond = options.variantCondition
    val collection = options.collection

    val vSymTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    // FIXME use custom parser with constraint on Solr field name
    val vparsed = Parser.parseAnnotationArgs(vCond, vEC)
      .map { case (name, t, f) =>
        assert(name.tail == Nil)
        (name.head, t, f)
      }

    val gSymTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas),
      "s" ->(2, TSample),
      "sa" ->(3, sas),
      "g" ->(4, TGenotype))
    val gEC = EvalContext(gSymTab)
    val gA = gEC.a

    val gparsed = Parser.parseAnnotationArgs(gCond, gEC)
      .map { case (name, t, f) =>
        assert(name.tail == Nil)
        (name.head, t, f)
      }

    val url = options.url
    val zkHost = options.zkHost

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

    // retrieve current fields
    val fieldsResponse = new SchemaRequest.Fields().process(solr)

    val preexistingFields = fieldsResponse.getFields.asScala
      .map(_.asScala("name").asInstanceOf[String])
      .toSet

    val addFieldReqs = vparsed.flatMap { case (name, t, f) =>
      addFieldReq(preexistingFields, name, t)
    } ++ vds.sampleIds.flatMap { s =>
      gparsed.flatMap { case (name, t, f) =>
        addFieldReq(preexistingFields, s + "_" + name, t)
      }
    }

    info(s"adding ${addFieldReqs.length} fields")

    if (addFieldReqs.nonEmpty) {
      val req = new SchemaRequest.MultiUpdate((addFieldReqs.toList: List[SchemaRequest.Update]).asJava)
      req.process(solr)
    }

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.sampleAnnotations)

    vds.rdd.foreachPartition { it =>
      val ab = mutable.ArrayBuilder.make[AnyRef]
      val documents = it.map { case (v, va, gs) =>

        val document = new SolrInputDocument()

        vparsed.foreach { case (name, t, f) =>
          vEC.setAll(v, va)
          f().foreach(x => documentAddField(document, name, t, x))
        }

        gs.iterator.zipWithIndex.foreach { case (g, i) =>
          if (g.isCalled && !g.isHomRef) {
            val s = sampleIdsBc.value(i)
            val sa = sampleAnnotationsBc.value(i)
            gparsed.foreach { case (name, t, f) =>
              gEC.setAll(v, va, s, sa, g)
              f().foreach(x => documentAddField(document, s + "_" + name, t, x))
            }
          }
        }

        document
      }

      val solr =
        if (url != null)
          new HttpSolrClient(url)
        else {
          val cc = new CloudSolrClient(zkHost)
          cc.setDefaultCollection(collection)
          cc
        }

      var retry = true
      var retryInterval = 3 * 1000 // 3s
      val maxRetryInterval = 3 * 60 * 1000 // 3m

      while (retry) {
        try {
          solr.add(documents.asJava)
          solr.commit()
          retry = false
        } catch {
          case e: SolrException =>
            warn("add documents timeout, retrying")

            Thread.sleep(Random.nextInt(retryInterval))
            retryInterval = (retryInterval * 2).max(maxRetryInterval)
        }
      }
    }

    state
  }
}

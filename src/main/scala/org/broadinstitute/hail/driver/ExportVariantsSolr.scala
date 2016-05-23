package org.broadinstitute.hail.driver

import org.apache.solr.client.solrj.impl.HttpSolrClient
import org.apache.solr.common.SolrInputDocument
import org.broadinstitute.hail.expr.{EvalContext, Parser, TArray, TVariant}

import scala.collection.JavaConverters._
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object ExportVariantsSolr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "comma-separated list of fields/computations to be exported")
    var condition: String = _

    @Args4jOption(required = true, name = "-u", aliases = Array("--url"),
      usage = "Solr instance (URL) to connect to")
    var url: String = _

  }

  def newOptions = new Options

  def name = "exportvariantssolr"

  def description = "Export variant information to Solr"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val sc = state.vds.sparkContext
    val vds = state.vds
    val vas = vds.vaSignature
    val cond = options.condition

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas))
    val ec = EvalContext(symTab)
    val a = ec.a

    val (header, fs) = Parser.parseExportArgs(cond, ec)
    if (header.isEmpty)
      fatal("column names required in condition")

    val columns = header.get.split("\t")
    val url = options.url

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    vds.rdd.foreachPartition { it =>
      val ab = mutable.ArrayBuilder.make[AnyRef]
      val documents = it.map { case (v, va, gs) =>

        val document = new SolrInputDocument()

        /*
        document.addField("chr", v.contig)
        document.addField("pos", v.start)
        document.addField("ref", v.ref)
        document.addField("pos", v.alt) */

        fs.zip(columns).foreach { case (f, col) =>
          a(0) = v
          a(1) = va
          document.addField(col, f().asInstanceOf[AnyRef])
        }

        gs.iterator.zip(sampleIdsBc.value.iterator).foreach { case (g, id) =>
          g.nNonRefAlleles
            .filter(_ > 0)
            .foreach { n =>
              document.addField(id + "_num_alt", n.toString)
              g.ad.foreach { ada =>
                val ab = ada(0).toDouble / (ada(0) + ada(1))
                document.addField(id + "_ab", ab.toString)
              }
              g.gq.foreach { gqx =>
                document.addField(id + "_gq", gqx.toString)
              }
            }
        }

        document
      }

      val solr = new HttpSolrClient(url)
      solr.add(documents.asJava)

      // FIXME back off it commit fails
      solr.commit()
    }

    state
  }
}

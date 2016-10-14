package org.broadinstitute.hail.driver

import com.datastax.driver.core._
import com.datastax.driver.core.querybuilder.QueryBuilder
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.StringEscapeUtils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.JavaConverters._
import scala.collection.mutable

// FIXME add drop/create table option

object CassandraStuff {
  private var cluster: Cluster = null
  private var session: Session = null

  private var refcount: Int = 0

  def getSession(address: String): Session = {
    this.synchronized {
      if (cluster == null)
        cluster = Cluster.builder()
          .addContactPoint(address)
          .build()

      if (session == null)
        session = cluster.connect()

      refcount += 1
    }

    session
  }

  def disconnect() {
    this.synchronized {
      refcount -= 1
      if (refcount == 0) {
        session.close()
        cluster.close()

        session = null
        cluster = null
      }
    }
  }
}

object ExportVariantsCass extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-a", aliases = Array("--address"),
      usage = "Cassandra contact point to connect to")
    var address: String = _

    @Args4jOption(name = "--export-ref", usage = "export HomRef calls")
    var exportRef = false

    @Args4jOption(name = "--export-missing", usage = "export missing genotypes")
    var exportMissing = false

    @Args4jOption(required = true, name = "-g",
      usage = "comma-separated list of fields/computations to be exported")
    var genotypeCondition: String = _

    @Args4jOption(required = true, name = "-k",
      usage = "Cassandra keyspace")
    var keyspace: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--table"),
      usage = "Cassandra table")
    var table: String = _

    @Args4jOption(required = true, name = "-v",
      usage = "comma-separated list of fields/computations to be exported")
    var variantCondition: String = _

  }

  def newOptions = new Options

  def name = "exportvariantscass"

  def description = "Export variant information to Cassandra"

  def supportsMultiallelic = true

  def requiresVDS = true

  def toCassType(t: Type): String = t match {
    case TBoolean => "boolean"
    case TInt => "int"
    case TLong => "bigint"
    case TFloat => "float"
    case TDouble => "double"
    case TString => "text"
    case TArray(elementType) => s"list<${toCassType(elementType)}>"
    case TSet(elementType) => s"set<${toCassType(elementType)}>"
    case _ =>
      fatal("unsupported type: $t")
  }

  def toCassValue(a: Any, t: Type): AnyRef = t match {
    case TArray(elementType) => a.asInstanceOf[Seq[_]].asJava
    case TSet(elementType) => a.asInstanceOf[Set[_]].asJava
    case _ => a.asInstanceOf[AnyRef]
  }

  def escapeString(name: String): String =
    escapeStringSimple(name, '_', !_.isLetter, !_.isLetterOrDigit)

  def escapeCassColumnName(name: String): String = {
    val sb = new StringBuilder

    if (name.head.isDigit)
      sb += 'x'

    name.foreach { c =>
      if (c.isLetterOrDigit)
        sb += c.toLower
      else
        sb += '_'
    }

    sb.result()
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = vds.sparkContext
    val vas = vds.vaSignature
    val sas = vds.saSignature
    val gCond = options.genotypeCondition
    val vCond = options.variantCondition
    val address = options.address
    val exportRef = options.exportRef
    val exportMissing = options.exportMissing

    val keyspace = options.keyspace
    val table = options.table
    val qualifiedTable = keyspace + "." + table

    val vSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    val vparsed = Parser.parseNamedArgs(vCond, vEC)

    val gSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas),
      "s" -> (2, TSample),
      "sa" -> (3, sas),
      "g" -> (4, TGenotype))
    val gEC = EvalContext(gSymTab)
    val gA = gEC.a

    val gparsed = Parser.parseNamedArgs(gCond, gEC)

    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas))
    val ec = EvalContext(symTab)
    val a = ec.a

    val fields = (vparsed.map { case (name, t, f) => (escapeString(name), t) }
      ++ vds.sampleIds.flatMap { s =>
      gparsed.map { case (name, t, f) => (escapeString(s) + "__" + escapeString(name), t) }
    })

    val session = CassandraStuff.getSession(address)

    val keyspaceMetadata = session.getCluster.getMetadata.getKeyspace(keyspace)
    if (keyspaceMetadata == null) {
       throw new IllegalArgumentException("keyspace not found: " + keyspace)
    }
    val tableMetadata = keyspaceMetadata.getTable(table)
    if (tableMetadata == null) {
       throw new IllegalArgumentException("table not found: " + table)
    }

    val preexistingFields = tableMetadata.getColumns.asScala.map(_.getName).toSet
    val toAdd = fields
      .filter { case (name, t) => !preexistingFields(name) }

    if (toAdd.nonEmpty) {
      session.execute(s"ALTER TABLE $qualifiedTable ADD (${
        toAdd.map { case (name, t) => s""""$name" ${toCassType(t)}""" }.mkString(",")
      })")
    }

    CassandraStuff.disconnect()

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.sampleAnnotations)

    val futures = vds.rdd
      .foreachPartition { it =>
        val session = CassandraStuff.getSession(address)

        val nb = mutable.ArrayBuilder.make[String]
        val vb = mutable.ArrayBuilder.make[AnyRef]

        val futures = it
          .map { case (v, (va, gs)) =>
            nb.clear()
            vb.clear()

            vparsed.foreach { case (name, t, f) =>
              vEC.setAll(v, va)
              f().foreach { a =>
                nb += escapeString(name)
                vb += toCassValue(a, t)
              }
            }

            gs.iterator.zipWithIndex.foreach { case (g, i) =>
              val s = sampleIdsBc.value(i)
              val sa = sampleAnnotationsBc.value(i)
              gparsed.foreach { case (name, t, f) =>
                if ((exportMissing || g.isCalled) && (exportRef || !g.isHomRef)) {
                  gEC.setAll(v, va, s, sa, g)
                  f().foreach { a =>
                    nb += '"' + escapeString(s) + "__" + escapeString(name) + '"'
                    vb += toCassValue(a, t)
                  }
                }
              }
            }

            val names = nb.result()
            val values = vb.result()

            session.executeAsync(QueryBuilder
              .insertInto(keyspace, table)
              .values(names, values))
          }

        futures.foreach(_.getUninterruptibly())

        CassandraStuff.disconnect()
      }

    state
  }

}

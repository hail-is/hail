package org.broadinstitute.hail.driver

import com.datastax.driver.core._
import com.datastax.driver.core.querybuilder.QueryBuilder
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.JavaConverters._

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

  def supportsMultiallelic = false

  def requiresVDS = true

  def toCassType(t: Type): String = t match {
    case TBoolean => "boolean"
    case TInt => "int"
    case TLong => "bigint"
    case TFloat => "float"
    case TDouble => "double"
    case TString => "text"
    case TArray(elementType) =>
      s"list<${toCassType(elementType)}>"
    case _ =>
      // FIXME
      fatal("")
  }

  def toCassValue(a: Any, t: Type): AnyRef = t match {
    case TArray(elementType) => a.asInstanceOf[Seq[_]].asJava
    case _ => a.asInstanceOf[AnyRef]
  }

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

    val keyspace = options.keyspace
    val table = options.table
    val qualifiedTable = keyspace + "." + table

    val vSymTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

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

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas))
    val ec = EvalContext(symTab)
    val a = ec.a

    val fields = (vparsed.map { case (name, t, f) => (escapeCassColumnName(name), t) }
      ++ vds.sampleIds.flatMap { s =>
      gparsed.map { case (name, t, f) => (escapeCassColumnName(s + "_" + name), t) }
    })

    val session = CassandraStuff.getSession(address)

    // FIXME check keyspace, table exsit (null)
    val tableMetadata = session.getCluster.getMetadata.getKeyspace(keyspace).getTable(table)
    val preexistingFields = tableMetadata.getColumns.asScala.map(_.getName).toSet
    val toAdd = fields
      .filter { case (name, t) => !preexistingFields(name) }

    if (toAdd.nonEmpty) {
      session.execute(s"ALTER TABLE $qualifiedTable ADD (${
        toAdd.map { case (name, t) => s"$name ${toCassType(t)}" }.mkString(",")
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
          .map { case (v, va, gs) =>
            nb.clear()
            vb.clear()

            vparsed.foreach { case (name, t, f) =>
              vEC.setAll(v, va)
              f().foreach { a =>
                nb += escapeCassColumnName(name)
                vb += toCassValue(a, t)
              }
            }

            gs.iterator.zipWithIndex.foreach { case (g, i) =>
              val s = sampleIdsBc.value(i)
              val sa = sampleAnnotationsBc.value(i)
              gparsed.foreach { case (name, t, f) =>
                if (g.isCalled && !g.isHomRef) {
                  gEC.setAll(v, va, s, sa, g)
                  f().foreach { a =>
                    nb += escapeCassColumnName(s + "_" + name)
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

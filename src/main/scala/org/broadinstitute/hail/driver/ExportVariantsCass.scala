package org.broadinstitute.hail.driver

import com.datastax.driver.core.{BoundStatement, Cluster, Session}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

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

  def releaseSession() {
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

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "comma-separated list of fields/computations to be exported")
    var condition: String = _

    @Args4jOption(required = true, name = "-a", aliases = Array("--address"),
      usage = "Cassandra contact point to connect to")
    var address: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--table"),
      usage = "Cassandra table to export to")
    var table: String = _

  }

  def newOptions = new Options

  def name = "exportvariantscass"

  def description = "Export variant information to Cassandra"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
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
    val address = options.address

    val query =
      s"""
         |INSERT INTO ${options.table}
         |(chrom, pos, ref, alt, ${columns.mkString(", ")})
         |VALUES (?, ?, ?, ?, ${columns.map(_ => "?").mkString(", ")});
      """.stripMargin

    vds.variantsAndAnnotations
      .foreachPartition { it =>
        val session = CassandraStuff.getSession(address)

        val statement = session.prepare(query)
        val ab = mutable.ArrayBuilder.make[AnyRef]

        val futures = it.map { case (v, va) =>
          ab.clear()
          ab += v.contig
          ab += v.start.asInstanceOf[AnyRef]
          ab += v.ref
          ab += v.alt
          fs.foreach { f =>
            a(0) = v
            a(1) = va
            ab += f().asInstanceOf[AnyRef]
          }
          val r = ab.result()
          val boundStatement = new BoundStatement(statement)

          session.executeAsync(boundStatement.bind(r: _*))
        }

        futures.foreach(_.getUninterruptibly())

        CassandraStuff.releaseSession()
      }

    state
  }
}

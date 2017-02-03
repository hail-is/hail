package is.hail.io

import com.datastax.driver.core.querybuilder.QueryBuilder
import com.datastax.driver.core.schemabuilder.SchemaBuilder
import com.datastax.driver.core.{Cluster, Session}
import is.hail.expr.{EvalContext, Parser, TArray, TBoolean, TDouble, TFloat, TGenotype, TInt, TLong, TSample, TSet, TString, TVariant, Type}
import is.hail.utils.StringEscapeUtils.escapeStringSimple
import is.hail.utils.{fatal, info, warn}
import is.hail.variant.VariantDataset

import scala.collection.JavaConverters._
import scala.collection.mutable

object CassandraConnector {
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

  def toCassType(t: Type): String = t match {
    case TBoolean => "boolean"
    case TInt => "int"
    case TLong => "bigint"
    case TFloat => "float"
    case TDouble => "double"
    case TString => "text"
    case TArray(elementType) => s"list<${ toCassType(elementType) }>"
    case TSet(elementType) => s"set<${ toCassType(elementType) }>"
    case _ =>
      fatal(s"unsupported type: $t")
  }

  def toCassValue(a: Option[Any], t: Type): AnyRef = t match {
    case TArray(elementType) => a.map(_.asInstanceOf[Seq[_]].asJava).orNull
    case TSet(elementType) => a.map(_.asInstanceOf[Set[_]].asJava).orNull
    case _ => a.map(_.asInstanceOf[AnyRef]).orNull
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

  def exportVariants(vds: VariantDataset,
    address: String,
    keySpace: String,
    table: String,
    genotypeExpr: String,
    variantExpr: String,
    drop: Boolean = false,
    exportRef: Boolean = false,
    exportMissing: Boolean = false,
    blockSize: Int = 100) {

    val sc = vds.sparkContext
    val vas = vds.vaSignature
    val sas = vds.saSignature

    val qualifiedTable = keySpace + "." + table

    val vSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    val (vNames, vTypes, vf) = Parser.parseNamedExprs(variantExpr, vEC)

    val gSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas),
      "s" -> (2, TSample),
      "sa" -> (3, sas),
      "g" -> (4, TGenotype))
    val gEC = EvalContext(gSymTab)
    val gA = gEC.a

    val (gHeader, gTypes, gf) = Parser.parseNamedExprs(genotypeExpr, gEC)

    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas))
    val ec = EvalContext(symTab)

    val fields = vNames.map(escapeString).zip(vTypes) ++ vds.sampleIds.flatMap { s =>
      gHeader.map(field => s"${ escapeString(s) }__${ escapeString(field) }").zip(gTypes)
    }

    val session = CassandraConnector.getSession(address)

    // get keyspace (create it if it doesn't exist)
    var keyspaceMetadata = session.getCluster.getMetadata.getKeyspace(keySpace)
    if (keyspaceMetadata == null) {
      info(s"creating keyspace ${ keySpace }")
      try {
        session.execute(s"CREATE KEYSPACE ${ keySpace } " +
          "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}  AND durable_writes = true")
      } catch {
        case e: Exception => fatal(s"exportvariantscass: unable to create keyspace ${ keySpace }: ${ e }")
      }
      keyspaceMetadata = session.getCluster.getMetadata.getKeyspace(keySpace)
    }

    // get table (drop and create it if necessary)
    if (drop) {
      info(s"dropping table ${ qualifiedTable }")
      try {
        session.execute(SchemaBuilder.dropTable(keySpace, table).ifExists());
      } catch {
        case e: Exception => warn(s"exportvariantscass: unable to drop table ${ qualifiedTable }: ${ e }")
      }
    }

    var tableMetadata = keyspaceMetadata.getTable(table)
    if (tableMetadata == null) {
      info(s"creating table ${ qualifiedTable }")
      try {
        session.execute(s"CREATE TABLE $qualifiedTable (${escapeString("dataset_id")} text, chrom text, start int, ref text, alt text, " +
          s"PRIMARY KEY ((${escapeString("dataset_id")}, chrom, start), ref, alt))") // WITH COMPACT STORAGE")
      } catch {
        case e: Exception => fatal(s"exportvariantscass: unable to create table ${ qualifiedTable }: ${ e }")
      }
      //info(s"table ${qualifiedTable} created")
      tableMetadata = keyspaceMetadata.getTable(table)
    }

    val preexistingFields = tableMetadata.getColumns.asScala.map(_.getName).toSet
    val toAdd = fields
      .filter { case (name, t) => !preexistingFields(name) }

    if (toAdd.nonEmpty) {
      session.execute(s"ALTER TABLE $qualifiedTable ADD (${
        toAdd.map { case (name, t) => s""""$name" ${ toCassType(t) }""" }.mkString(",")
      })")
    }

    CassandraConnector.disconnect()

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.sampleAnnotations)
    val localBlockSize = blockSize

    val futures = vds.rdd
      .foreachPartition { it =>
        val session = CassandraConnector.getSession(address)
        val nb = mutable.ArrayBuilder.make[String]
        val vb = mutable.ArrayBuilder.make[AnyRef]

        it
          .grouped(localBlockSize)
          .foreach { block =>
            val futures = block
              .map { case (v, (va, gs)) =>
                nb.clear()
                vb.clear()

                vEC.setAll(v, va)
                vf().zipWithIndex.foreach { case (a, i) =>
                  nb += s""""${ escapeString(vNames(i)) }""""
                  vb += toCassValue(a, vTypes(i))
                }

                gs.iterator.zipWithIndex.foreach { case (g, i) =>
                  val s = sampleIdsBc.value(i)
                  val sa = sampleAnnotationsBc.value(i)
                  if ((exportMissing || g.isCalled) && (exportRef || !g.isHomRef)) {
                    gEC.setAll(v, va, s, sa, g)
                    gf().zipWithIndex.foreach { case (a, j) =>
                      nb += s""""${ escapeString(s) }__${ escapeString(gHeader(j)) }""""
                      vb += toCassValue(a, gTypes(j))
                    }
                  }
                }

                val names = nb.result()
                val values = vb.result()

                session.executeAsync(QueryBuilder
                  .insertInto(keySpace, table)
                  .values(names, values))
              }

            futures.foreach(_.getUninterruptibly())
          }

        CassandraConnector.disconnect()
      }
  }
}

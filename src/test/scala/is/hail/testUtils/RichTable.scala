package is.hail.testUtils

import is.hail.annotations.Inserter
import is.hail.expr._
import is.hail.expr.ir.{InsertFields, MakeStruct, Ref}
import is.hail.expr.types.TStruct
import is.hail.table.Table
import is.hail.utils._
import org.apache.spark.sql.Row

class RichTable(ht: Table) {
  def forall(code: String): Boolean = {
    val ec = ht.rowEvalContext()
    ec.set(0, ht.globals.value)

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    ht.rdd.forall { a =>
      ec.set(1, a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def exists(code: String): Boolean = {
    val ec = ht.rowEvalContext()
    ec.set(0, ht.globals.value)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    ht.rdd.exists { a =>
      ec.set(1, a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def rename(rowUpdateMap: Map[String, String], globalUpdateMap: Map[String, String]): Table = {
    ht.select(
      "{" + ht.fieldNames.map(n => s"${ rowUpdateMap.getOrElse(n, n) }: row.$n").mkString(",") + "}",
      ht.key.map(_.map(k => rowUpdateMap.getOrElse(k, k)).toFastIndexedSeq),
      ht.key.map(_.length))
      .selectGlobal(
        "{" + ht.globalSignature.fieldNames.map(n => s"${ globalUpdateMap.getOrElse(n, n) } = global.$n").mkString(",") + "}")
  }

  def keyByExpr(exprs: (String, String)*): Table =
    ht.select(s"annotate(row, {${ exprs.map { case (n, e) => s"${ prettyIdentifier(n) }: $e" }.mkString(",") }})", Some(exprs.map(_._1).toIndexedSeq), Some(0))

  def annotate(exprs: (String, String)*): Table =
    ht.select(s"annotate(row, {${ exprs.map { case (n, e) => s"${ prettyIdentifier(n) }: $e" }.mkString(",") }})", ht.key, ht.key.map(_.length))
}

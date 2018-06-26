package is.hail.testUtils

import is.hail.annotations.Inserter
import is.hail.expr._
import is.hail.expr.ir.{InsertFields, MakeStruct, Ref}
import is.hail.expr.types.{TBoolean, TStruct}
import is.hail.table.Table
import is.hail.utils._
import org.apache.spark.sql.Row

class RichTable(ht: Table) {
  def forall(code: String): Boolean = {
    val (a, t) = ht.aggregate(s"AGG.filter(row => let t = $code in !(isDefined(t) && t)).count() == i64#0")
    assert(t.isOfType(TBoolean()))
    assert(a != null)
    a.asInstanceOf[Boolean]
  }

  def exists(code: String): Boolean = {
    val (a, t) = ht.aggregate(s"AGG.filter(row => let t = $code in isDefined(t) && t).count() > i64#0")
    assert(t.isOfType(TBoolean()))
    assert(a != null)
    a.asInstanceOf[Boolean]
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

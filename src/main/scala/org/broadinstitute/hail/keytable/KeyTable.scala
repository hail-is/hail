package org.broadinstitute.hail.keytable

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{BaseType, EvalContext, Parser, TBoolean, TStruct}
import org.broadinstitute.hail.methods.Filter

case class KeyTable(rdd: RDD[Annotation], signature: TStruct, keyNames: Array[String]) {

  val fieldNames = signature.fields.map(_.name)

  require(fieldNames.distinct.length == fieldNames.length)
  require(keyNames.forall(k => signature.selfField(k).isDefined))

  def nRows = rdd.count()

  def leftJoin(other: KeyTable, joinKeys: Array[String]): KeyTable = ???
  def rightJoin(other: KeyTable, joinKeys: Array[String]): KeyTable = ???
  def outerJoin(other: KeyTable): KeyTable = ???
  def innerJoin(other: KeyTable): KeyTable = ???
//    require(keyNames.toSet == other.keyNames.toSet)
    // function to make key order the same


  def query(code: String): (BaseType, Querier) = {
    val ec = EvalContext(signature.fields.map(f => (f.name, f.`type`)): _*)

    val (t, f) = Parser.parse(code, ec)

    val f2: Annotation => Option[Any] = { annotation =>
      Option(annotation).map(_.asInstanceOf[Row]) match {
        case Some(r) => ec.setAll(r.toSeq: _*)
        case None => ec.setAll(Seq.fill(signature.size)(null))
      }
      f()
    }

    (t, f2)
  }

  def filter(p: (Annotation) => Boolean): KeyTable = copy(rdd = rdd.filter { a => p(a) })

  def filterExpr(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(signature.fields.map(f => (f.name, f.`type`)): _*)

    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

    val p = (a: Annotation) => {
      Option(a).map(_.asInstanceOf[Row]) match {
        case Some(r) => ec.setAll(r.toSeq: _*)
        case None => ec.setAll(Seq.fill(signature.size)(null))
      }

      Filter.keepThis(f(), keep)
    }

    filter(p)
  }
}

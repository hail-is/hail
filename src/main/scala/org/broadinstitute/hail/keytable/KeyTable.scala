package org.broadinstitute.hail.keytable

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{BaseType, EvalContext, Parser, TBoolean, TStruct}
import org.broadinstitute.hail.methods.Filter

case class KeyTable (rdd: RDD[(Annotation, Annotation)], keySignature: TStruct, valueSignature: TStruct) {

  val fieldNames = (keySignature.fields ++ valueSignature.fields).map(_.name)

  assert(fieldNames.distinct.length == fieldNames.length)

  def queryKey(code: String): (BaseType, Querier) = ???

  def queryValue(code: String): (BaseType, Querier) = ???

  def filter(p: (Annotation, Annotation) => Boolean): KeyTable = copy(rdd = rdd.filter{ case (k, v) => p(k, v)})

  def filterRowsExpr(cond: String, keep: Boolean): KeyTable = {
    val symTab = (keySignature.fields ++ valueSignature.fields)
      .zipWithIndex.map{case (fd, i) => (fd.name, (i, fd.`type`))}.toMap

    val ec = EvalContext(symTab)

    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

    val p = (k: Annotation, v: Annotation) => {
      ec.setAll(k, v)
      Filter.keepThis(f(), keep)
    }

    filter(p)
  }
}

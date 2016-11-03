package org.broadinstitute.hail.keytable

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{BaseType, EvalContext, Parser, TBoolean, TStruct}
import org.broadinstitute.hail.methods.Filter
import org.broadinstitute.hail.utils._

object KeyTable extends Serializable {
  def setEvalContext(ec: EvalContext, k: Annotation, v: Annotation, nKeys: Int) = {
    (Option(k).map(_.asInstanceOf[Row]), Option(v).map(_.asInstanceOf[Row])) match {
      case (Some(kr), Some(vr)) => ec.setAll(kr.toSeq ++ vr.toSeq: _*)
      case (Some(kr), None) =>
        ec.clear()
        ec.setAll(kr.toSeq: _*)
      case (None, Some(vr)) =>
        ec.clear()
        vr.toSeq.zipWithIndex.foreach{ case (a, i) => ec.set(i + nKeys, a)}
      case (None, None) => ec.clear()
    }
  }

  def setEvalContext(ec: EvalContext, a: Annotation) = {
    Option(a).map(_.asInstanceOf[Row]) match {
      case Some(r) => ec.setAll(r.toSeq: _*)
      case _ => ec.clear()
    }
  }

  def pairSignature(signature: TStruct, keyNames: Array[String]): (TStruct, TStruct) = {
    val keyNameSet = keyNames.toSet
    (TStruct(signature.fields.filter(fd => keyNameSet.contains(fd.name))),
      TStruct(signature.fields.filterNot(fd => keyNameSet.contains(fd.name))))
  }

  def singleSignature(keySignature: TStruct, valueSignature: TStruct): (TStruct, Array[String]) =
    (TStruct(keySignature.fields ++ valueSignature.fields), keySignature.fields.map(_.name).toArray)

  def toSingleRDD(rdd: RDD[(Annotation, Annotation)], keySignature: TStruct, valueSignature: TStruct): RDD[Annotation] =
    rdd.map{ case (k, v) => Annotation(Option(k).map(_.asInstanceOf[Row]).toSeq ++ Option(v).map(_.asInstanceOf[Row]).toSeq: _*) }

  def toPairRDD(rdd: RDD[Annotation], signature: TStruct, keyNames: Array[String]): RDD[(Annotation, Annotation)] = {
    val keyNameSet = keyNames.toSet
    val keyIndices = signature.fields.filter(fd => keyNames.contains(fd.name)).map(_.index).toSet
    val valueIndices = signature.fields.filterNot(fd => keyNames.contains(fd.name)).map(_.index).toSet

    rdd.map { a =>
      val r = Option(a).map(_.asInstanceOf[Row].toSeq).getOrElse(Seq.fill(signature.size)(null)).zipWithIndex
      val keyRow = r.filter{ case (ann, i) => keyIndices.contains(i) }.map(_._1)
      val valueRow = r.filter{ case (ann, i) => valueIndices.contains(i) }.map(_._1)
      (Annotation.fromSeq(keyRow), Annotation.fromSeq(valueRow))
    }
  }

  def apply(rdd: RDD[Annotation], signature: TStruct, keyNames: Array[String]): KeyTable = {
    val (keySignature, valueSignature) = pairSignature(signature, keyNames)
    KeyTable(toPairRDD(rdd, signature, keyNames), keySignature, valueSignature)
  }
}

case class KeyTable(rdd: RDD[(Annotation, Annotation)], keySignature: TStruct, valueSignature: TStruct) {

  require(fieldNames.toSet.size == fieldNames.length)

  def signature = KeyTable.singleSignature(keySignature, valueSignature)._1
  def fields = signature.fields

  def keySchema = keySignature.schema
  def valueSchema = valueSignature.schema
  def schema = signature.schema

  def keyNames = keySignature.fields.map(_.name)
  def valueNames = valueSignature.fields.map(_.name)
  def fieldNames = keyNames ++ valueNames

  def nRows = rdd.count()
  def nFields = fields.length
  def nKeys = keySignature.size
  def nValues = valueSignature.size

  def leftJoin(other: KeyTable): KeyTable = ???
  def rightJoin(other: KeyTable): KeyTable = ???
  def outerJoin(other: KeyTable): KeyTable = ???
  def innerJoin(other: KeyTable): KeyTable = ???

//    require(keyNames.toSet == other.keyNames.toSet)
    // function to make key order the same

//  def mapAnnotations(f: (Annotation) => Annotation): KeyTable =
//    copy(rdd = KeyTable.toSingleRDD(rdd).map{ a => f(a)})

  def mapAnnotations(f: (Annotation, Annotation) => Annotation): KeyTable =
    copy(rdd = rdd.mapValuesWithKey{ case (k, v) => f(k, v) })

  def query(code: String): (BaseType, (Annotation, Annotation) => Option[Any]) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)

    val (t, f) = Parser.parse(code, ec)

    val f2: (Annotation, Annotation) => Option[Any] = { case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeys)
      f()
    }

    (t, f2)
  }

//  def query(code: String): (BaseType, Querier) = {
//    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
//
//    val (t, f) = Parser.parse(code, ec)
//
//    val f2: (Annotation) => Option[Any] = { a =>
//      KeyTable.setEvalContext(ec, a)
//      f()
//    }
//
//    (t, f2)
//  }

  def filter(p: (Annotation, Annotation) => Boolean): KeyTable = copy(rdd = rdd.filter { case (k, v) => p(k, v) })

  def filterExpr(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)

    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

    val p = (k: Annotation, v: Annotation) => {
      KeyTable.setEvalContext(ec, k, v, nKeys)
      Filter.keepThis(f(), keep)
    }

    filter(p)
  }
}

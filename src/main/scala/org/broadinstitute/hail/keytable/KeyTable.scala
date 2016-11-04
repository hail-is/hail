package org.broadinstitute.hail.keytable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.expr.{BaseType, EvalContext, Parser, TBoolean, TStruct, Type}
import org.broadinstitute.hail.methods.Filter
import org.broadinstitute.hail.utils._

object KeyTable extends Serializable {
  def annotationToSeq(a: Annotation, nFields: Int) = Option(a).map(_.asInstanceOf[Row].toSeq).getOrElse(Seq.fill[Any](nFields)(null))

  def setEvalContext(ec: EvalContext, k: Annotation, v: Annotation, nKeys: Int, nValues: Int) =
    ec.setAll(annotationToSeq(k, nKeys) ++ annotationToSeq(v, nValues): _*)

  def setEvalContext(ec: EvalContext, a: Annotation, nFields: Int) =
    ec.setAll(annotationToSeq(a, nFields): _*)

  def pairSignature(signature: TStruct, keyNames: Array[String]): (TStruct, TStruct) = {
    val keyNameSet = keyNames.toSet
    (TStruct(signature.fields.filter(fd => keyNameSet.contains(fd.name))),
      TStruct(signature.fields.filterNot(fd => keyNameSet.contains(fd.name))))
  }

  def singleSignature(keySignature: TStruct, valueSignature: TStruct): (TStruct, Array[String]) =
    (TStruct(keySignature.fields ++ valueSignature.fields), keySignature.fields.map(_.name).toArray)

  def toSingleRDD(rdd: RDD[(Annotation, Annotation)], nKeys: Int, nValues: Int): RDD[Annotation] =
    rdd.map{ case (k, v) =>
      val x = Annotation.fromSeq(annotationToSeq(k, nKeys) ++ annotationToSeq(v, nValues))
      x
    }

  def toPairRDD(rdd: RDD[Annotation], signature: TStruct, keyNames: Array[String]): RDD[(Annotation, Annotation)] = {
    val keyNameSet = keyNames.toSet
    val keyIndices = signature.fields.filter(fd => keyNames.contains(fd.name)).map(_.index).toSet
    val valueIndices = signature.fields.filterNot(fd => keyNames.contains(fd.name)).map(_.index).toSet
    val nFields = signature.size

    rdd.map { a =>
      val r = annotationToSeq(a, nFields).zipWithIndex
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

  def same(other: KeyTable): Boolean = {
    if (fields.toSet != other.fields.toSet) {
      println(s"signature: this=${ schema } other=${ other.schema }")
      false
    } else if (keyNames.toSet != other.keyNames.toSet) {
      println(s"keyNames: this=${ keyNames.mkString(",") } other=${ other.keyNames.mkString(",")}")
      false
    } else {
      val thisFieldNames = valueNames
      val otherFieldNames = other.valueNames

      rdd.groupByKey().fullOuterJoin(other.rdd.groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (None, None) => true
          case (Some(x), Some(y)) =>
            val r1 = x.map(r => thisFieldNames.zip(r.asInstanceOf[Row].toSeq).toMap).toSet
            val r2 = y.map(r => otherFieldNames.zip(r.asInstanceOf[Row].toSeq).toMap).toSet
            val res = r1 == r2
            if (!res)
              println(s"k=$k r1=${r1.mkString(",")} r2=${r2.mkString(",")}")
            res
          case _ =>
            println(s"k=$k v1=$v1 v2=$v2")
            false
        }
      }
    }
  }

  def mapAnnotations(f: (Annotation) => Annotation, newSignature: TStruct, newKeyNames: Array[String]): KeyTable =
    KeyTable(KeyTable.toSingleRDD(rdd, nKeys, nValues).map(a => f(a)), newSignature, newKeyNames)

  def mapAnnotations(f: (Annotation, Annotation) => Annotation, newValueSignature: TStruct): KeyTable =
    copy(rdd = rdd.mapValuesWithKey{ case (k, v) => f(k, v) }, valueSignature = newValueSignature)

  def query(code: String): (BaseType, (Annotation, Annotation) => Option[Any]) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)

    val (t, f) = Parser.parse(code, ec)

    val f2: (Annotation, Annotation) => Option[Any] = { case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeys, nValues)
      f()
    }

    (t, f2)
  }

  def querySingle(code: String): (BaseType, Querier) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)

    val (t, f) = Parser.parse(code, ec)

    val f2: (Annotation) => Option[Any] = { a =>
      KeyTable.setEvalContext(ec, a, nFields)
      f()
    }

    (t, f2)
  }

  def filter(p: (Annotation, Annotation) => Boolean): KeyTable = copy(rdd = rdd.filter { case (k, v) => p(k, v) })

  def filterExpr(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)

    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

    val p = (k: Annotation, v: Annotation) => {
      KeyTable.setEvalContext(ec, k, v, nKeys, nValues)
      Filter.keepThis(f(), keep)
    }

    filter(p)
  }


  def leftJoin(other: KeyTable, joinKeys: Array[String]): KeyTable = {
     keySignature.merge(valueSignature)
  }

  def rightJoin(other: KeyTable, joinKeys: Array[String]): KeyTable = ???
  def outerJoin(other: KeyTable, joinKeys: Array[String]): KeyTable = ???
  def innerJoin(other: KeyTable, joinKeys: Array[String]): KeyTable = ???

  //    require(keyNames.toSet == other.keyNames.toSet)
  // function to make key order the same
}
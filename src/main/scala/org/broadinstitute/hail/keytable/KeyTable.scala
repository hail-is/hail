package org.broadinstitute.hail.keytable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.{Aggregators, Filter}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.TextExporter

import scala.collection.mutable
import scala.reflect.ClassTag


object KeyTable extends Serializable with TextExporter {

  def importTextTable(sc: SparkContext, path: Array[String], keysStr: String, nPartitions: Int, config: TextTableConfiguration) = {
    require(nPartitions > 1)

    val files = sc.hadoopConfiguration.globAll(path)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val keys = Parser.parseIdentifierList(keysStr)

    val (struct, rdd) =
      TextTableReader.read(sc)(files, config, nPartitions)

    val invalidKeys = keys.filter(!struct.hasField(_))
    if (invalidKeys.nonEmpty)
      fatal(s"invalid keys: ${ invalidKeys.mkString(", ") }")

    KeyTable(rdd.map(_.value), struct, keys)
  }

  def annotationToSeq(a: Annotation, nFields: Int) = Option(a).map(_.asInstanceOf[Row].toSeq).getOrElse(Seq.fill[Any](nFields)(null))

  def setEvalContext(ec: EvalContext, k: Annotation, v: Annotation, nKeys: Int, nValues: Int) =
    ec.setAll(annotationToSeq(k, nKeys) ++ annotationToSeq(v, nValues): _*)

  def setEvalContext(ec: EvalContext, a: Annotation, nFields: Int) =
    ec.setAll(annotationToSeq(a, nFields): _*)

  def toSingleRDD(rdd: RDD[(Annotation, Annotation)], nKeys: Int, nValues: Int): RDD[Annotation] =
    rdd.map { case (k, v) =>
      val x = Annotation.fromSeq(annotationToSeq(k, nKeys) ++ annotationToSeq(v, nValues))
      x
    }

  def apply(rdd: RDD[Annotation], signature: TStruct, keyNames: Array[String]): KeyTable = {
    val keyFields = signature.fields.filter(fd => keyNames.contains(fd.name))
    val keyIndices = keyFields.map(_.index)

    val valueFields = signature.fields.filterNot(fd => keyNames.contains(fd.name))
    val valueIndices = valueFields.map(_.index)

    assert(keyIndices.toSet.intersect(valueIndices.toSet).isEmpty)

    val nFields = signature.size

    val newKeySignature = TStruct(keyFields.map(fd => (fd.name, fd.`type`)): _*)
    val newValueSignature = TStruct(valueFields.map(fd => (fd.name, fd.`type`)): _*)

    val newRDD = rdd.map { a =>
      val r = annotationToSeq(a, nFields).zipWithIndex
      val keyRow = keyIndices.map(i => r(i)._1)
      val valueRow = valueIndices.map(i => r(i)._1)
      (Annotation.fromSeq(keyRow), Annotation.fromSeq(valueRow))
    }

    KeyTable(newRDD, newKeySignature, newValueSignature)
  }
}

case class KeyTable(rdd: RDD[(Annotation, Annotation)], keySignature: TStruct, valueSignature: TStruct) {
  require(fieldNames.areDistinct())

  def signature = keySignature.merge(valueSignature)._1

  def fields = signature.fields

  def keySchema = keySignature.schema

  def valueSchema = valueSignature.schema

  def schema = signature.schema

  def keyNames = keySignature.fields.map(_.name).toArray

  def valueNames = valueSignature.fields.map(_.name).toArray

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
      println(s"keyNames: this=${ keyNames.mkString(",") } other=${ other.keyNames.mkString(",") }")
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
              println(s"k=$k r1=${ r1.mkString(",") } r2=${ r2.mkString(",") }")
            res
          case _ =>
            println(s"k=$k v1=$v1 v2=$v2")
            false
        }
      }
    }
  }

  def mapAnnotations[T](f: (Annotation) => T)(implicit tct: ClassTag[T]): RDD[T] =
    KeyTable.toSingleRDD(rdd, nKeys, nValues).map(a => f(a))

  def mapAnnotations[T](f: (Annotation, Annotation) => T)(implicit tct: ClassTag[T]): RDD[T] =
    rdd.map { case (k, v) => f(k, v) }

  def query(code: String): (BaseType, (Annotation, Annotation) => Option[Any]) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val (t, f) = Parser.parse(code, ec)

    val f2: (Annotation, Annotation) => Option[Any] = {
      case (k, v) =>
        KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
        f()
    }

    (t, f2)
  }

  def querySingle(code: String): (BaseType, Querier) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nFieldsLocal = nFields

    val (t, f) = Parser.parse(code, ec)

    val f2: (Annotation) => Option[Any] = { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)
      f()
    }

    (t, f2)
  }

  def annotate(cond: String, keyNameString: String): KeyTable = {
    val ec = EvalContext(fields.map(fd => (fd.name, fd.`type`)): _*)

    val (parseTypes, fns) = Parser.parseAnnotationArgs(cond, ec, None)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalSignature = parseTypes.foldLeft(signature) { case (vs, (ids, signature)) =>
      val (s: TStruct, i) = vs.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val keyNameArray = if (keyNameString != null) Parser.parseIdentifierList(keyNameString) else keyNames

    val nFieldsLocal = nFields

    val f: Annotation => Annotation = { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)

      fns.zip(inserters)
        .foldLeft(a) { case (a1, (fn, inserter)) =>
          inserter(a1, Option(fn()))
        }
    }

    KeyTable(mapAnnotations(f), finalSignature, keyNameArray)
  }

  def filter(p: (Annotation, Annotation) => Boolean): KeyTable =
    copy(rdd = rdd.filter { case (k, v) => p(k, v) })

  def filter(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

    val p = (k: Annotation, v: Annotation) => {
      KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
      Filter.keepThis(f(), keep)
    }

    filter(p)
  }

  def join(other: KeyTable, joinType: String): KeyTable = {
    if (keySignature != other.keySignature)
      fatal(
        s"""Key signatures must be identical.
            |Left signature: ${ keySignature.toPrettyString(compact = true) }
            |Right signature: ${ other.keySignature.toPrettyString(compact = true) }""".stripMargin)

    val overlappingFields = valueNames.toSet.intersect(other.valueNames.toSet)
    if (overlappingFields.nonEmpty)
      fatal(
        s"""Fields that are not keys cannot be present in both key-tables.
            |Overlapping fields: ${ overlappingFields.mkString(", ") }""".stripMargin)

    joinType match {
      case "left" => leftJoin(other)
      case "right" => rightJoin(other)
      case "inner" => innerJoin(other)
      case "outer" => outerJoin(other)
      case _ => fatal("Invalid join type specified. Choose one of `left', `right', `inner', `outer'")
    }
  }

  def leftJoin(other: KeyTable): KeyTable = {
    require(keySignature == other.keySignature)

    val (newValueSignature, merger) = valueSignature.merge(other.valueSignature)
    val newRDD = rdd.leftOuterJoin(other.rdd).map { case (k, (vl, vr)) => (k, merger(vl, vr.orNull)) }

    KeyTable(newRDD, keySignature, newValueSignature)
  }

  def rightJoin(other: KeyTable): KeyTable = {
    require(keySignature == other.keySignature)

    val (newValueSignature, merger) = valueSignature.merge(other.valueSignature)
    val newRDD = rdd.rightOuterJoin(other.rdd).map { case (k, (vl, vr)) => (k, merger(vl.orNull, vr)) }

    KeyTable(newRDD, keySignature, newValueSignature)
  }

  def outerJoin(other: KeyTable): KeyTable = {
    require(keySignature == other.keySignature)

    val (newValueSignature, merger) = valueSignature.merge(other.valueSignature)
    val newRDD = rdd.fullOuterJoin(other.rdd).map { case (k, (vl, vr)) => (k, merger(vl.orNull, vr.orNull)) }

    KeyTable(newRDD, keySignature, newValueSignature)
  }

  def innerJoin(other: KeyTable): KeyTable = {
    require(keySignature == other.keySignature)

    val (newValueSignature, merger) = valueSignature.merge(other.valueSignature)
    val newRDD = rdd.join(other.rdd).map { case (k, (vl, vr)) => (k, merger(vl, vr)) }

    KeyTable(newRDD, keySignature, newValueSignature)
  }

  def forall(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val f: () => Option[Boolean] = Parser.parse[Boolean](code, ec, TBoolean)

    rdd.forall { case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
      f().getOrElse(false)
    }
  }

  def exists(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val f: () => Option[Boolean] = Parser.parse[Boolean](code, ec, TBoolean)

    rdd.exists { case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
      f().getOrElse(false)
    }
  }

  def export(sc: SparkContext, output: String, typesFile: String) = {
    val hConf = sc.hadoopConfiguration

    val ec = EvalContext(fields.map(fd => (fd.name, fd.`type`)): _*)

    val (header, types, f) = Parser.parseNamedArgs(fieldNames.map(n => n + " = " + n).mkString(","), ec)

    Option(typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)

      KeyTable.exportTypes(file, hConf, typeInfo)
    }

    hConf.delete(output, recursive = true)

    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (k, v) =>
          sb.clear()
          KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
      }.writeTable(output, header.map(_.mkString("\t")))
  }

  def aggregate(keyCond: String, aggCond: String): KeyTable = {

    val aggregationEC = EvalContext(fields.map(fd => (fd.name, fd.`type`)): _*)
    val ec = EvalContext(fields.zipWithIndex.map { case (fd, i) => (fd.name, (-1, KeyTableAggregable(aggregationEC, fd.`type`, i))) }.toMap)

    val (keyNameParseTypes, keyF) =
      if (keyCond != null)
        Parser.parseAnnotationArgs(keyCond, aggregationEC, None)
      else
        (Array.empty[(List[String], Type)], Array.empty[() => Any])

    val (aggNameParseTypes, aggF) = Parser.parseAnnotationArgs(aggCond, ec, None)

    val keyNames = keyNameParseTypes.map(_._1.head)
    val aggNames = aggNameParseTypes.map(_._1.head)

    val keySignature = TStruct(keyNameParseTypes.map { case (n, t) => (n.head, t) }: _*)
    val valueSignature = TStruct(aggNameParseTypes.map { case (n, t) => (n.head, t) }: _*)

    val (zVals, _, combOp, resultOp) = Aggregators.makeFunctions(aggregationEC)
    val aggFunctions = aggregationEC.aggregationFunctions.map(_._1)

    assert(zVals.length == aggFunctions.length)

    val seqOp = (array: Array[Aggregator], b: Any) => {
      KeyTable.setEvalContext(aggregationEC, b, nFields)
      for (i <- array.indices) {
        array(i).seqOp(aggFunctions(i)(b))
      }
      array
    }

    val nFieldsLocal = nFields

    val newRDD = KeyTable.toSingleRDD(rdd, nKeys, nValues).mapPartitions { it =>
      it.map { a =>
        KeyTable.setEvalContext(aggregationEC, a, nFieldsLocal)
        val key = Annotation.fromSeq(keyF.map(_ ()))
        (key, a)
      }
    }.aggregateByKey(zVals)(seqOp, combOp)
      .map { case (k, agg) =>
        resultOp(agg)
        (k, Annotation.fromSeq(aggF.map(_ ())))
      }

    KeyTable(newRDD, keySignature, valueSignature)
  }
}
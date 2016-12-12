package org.broadinstitute.hail.keytable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.{Aggregators, Filter}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.TextExporter

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.collection.JavaConverters._


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

    assert(keyIndices.intersect(valueIndices).isEmpty)

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

  def fromDF(df: DataFrame, keyNames: Array[String]): KeyTable = {
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    KeyTable(df.rdd.map { r =>
      SparkAnnotationImpex.importAnnotation(r, signature)
    },
      signature, keyNames)
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

  def query(code: String): (Type, (Annotation, Annotation) => Option[Any]) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: (Annotation, Annotation) => Option[Any] = {
      case (k, v) =>
        KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
        f()
    }

    (t, f2)
  }

  def querySingle(code: String): (Type, Querier) = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nFieldsLocal = nFields

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: (Annotation) => Option[Any] = { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)
      f()
    }

    (t, f2)
  }

  def annotate(cond: String, keysStr: String): KeyTable = {
    val ec = EvalContext(fields.map(fd => (fd.name, fd.`type`)): _*)

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, None)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalSignature = (paths, types).zipped.foldLeft(signature) { case (vs, (ids, signature)) =>
      val (s: TStruct, i) = vs.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val keys = Parser.parseIdentifierList(keysStr)

    val nFieldsLocal = nFields

    val annotF: Annotation => Annotation = { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)

      f().zip(inserters)
        .foldLeft(a) { case (a1, (v, inserter)) =>
          inserter(a1, v)
        }
    }

    KeyTable(mapAnnotations(annotF), finalSignature, keys)
  }

  def filter(p: (Annotation, Annotation) => Boolean): KeyTable =
    copy(rdd = rdd.filter { case (k, v) => p(k, v) })

  def filter(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](cond, ec)

    val p = (k: Annotation, v: Annotation) => {
      KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
      Filter.keepThis(f(), keep)
    }

    filter(p)
  }

  def select(fieldsSelect: Array[String], newKeys: Array[String]): KeyTable = {
    val keyNamesNotInSelectedFields = newKeys.diff(fieldsSelect)
    if (keyNamesNotInSelectedFields.nonEmpty)
      fatal(s"Key fields `${ keyNamesNotInSelectedFields.mkString(", ") }' must be present in selected fields.")

    val fieldsNotExist = fieldsSelect.diff(fieldNames)
    if (fieldsNotExist.nonEmpty)
      fatal(s"Selected fields `${ fieldsNotExist.mkString(", ") }' do not exist in KeyTable. Choose from `${ fieldNames.mkString(", ") }'.")

    val fieldTransform = fieldsSelect.map(cn => fieldNames.indexOf(cn))
    val newSignature = TStruct(fieldTransform.map { i => signature.fields(i) })
    val nFieldsLocal = nFields

    val selectF: Annotation => Annotation = { a =>
      val row = KeyTable.annotationToSeq(a, nFieldsLocal)
      Annotation.fromSeq(fieldTransform.map(i => row(i)))
    }

    KeyTable(mapAnnotations(selectF), newSignature, newKeys)
  }

  def select(fieldsSelect: java.util.ArrayList[String], newKeys: java.util.ArrayList[String]): KeyTable =
    select(fieldsSelect.asScala.toArray, newKeys.asScala.toArray)

  def rename(fieldNameMap: Map[String, String]): KeyTable = {
    val newKeySignature = TStruct(keySignature.fields.map { fd => fd.copy(name = fieldNameMap.getOrElse(fd.name, fd.name)) })
    val newValueSignature = TStruct(valueSignature.fields.map { fd => fd.copy(name = fieldNameMap.getOrElse(fd.name, fd.name)) })

    val newFieldNames = newKeySignature.fields.map(_.name) ++ newValueSignature.fields.map(_.name)
    val duplicateFieldNames = newFieldNames.foldLeft(Map[String, Int]() withDefaultValue 0) { (m, x) => m + (x -> (m(x) + 1)) }.filter {
      _._2 > 1
    }

    if (duplicateFieldNames.nonEmpty)
      fatal(s"Found duplicate field names after renaming fields: `${ duplicateFieldNames.keys.mkString(", ") }'")

    KeyTable(rdd, newKeySignature, newValueSignature)
  }

  def rename(newFieldNames: Array[String]): KeyTable = {
    if (newFieldNames.length != nFields)
      fatal(s"Found ${ newFieldNames.length } new field names but need $nFields.")

    rename((fieldNames, newFieldNames).zipped.toMap)
  }

  def rename(fieldNameMap: java.util.HashMap[String, String]): KeyTable = rename(fieldNameMap.asScala.toMap)

  def rename(newFieldNames: java.util.ArrayList[String]): KeyTable = rename(newFieldNames.asScala.toArray)

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

    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](code, ec)

    rdd.forall { case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
      f().getOrElse(false)
    }
  }

  def exists(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.`type`)): _*)
    val nKeysLocal = nKeys
    val nValuesLocal = nValues

    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](code, ec)

    rdd.exists { case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeysLocal, nValuesLocal)
      f().getOrElse(false)
    }
  }

  def export(sc: SparkContext, output: String, typesFile: String) {
    val hConf = sc.hadoopConfiguration

    val ec = EvalContext(fields.map(fd => (fd.name, fd.`type`)): _*)

    // FIXME
    val (names, types, f) = Parser.parseExportExprs(fieldNames.map(n => n + " = " + n).mkString(","), ec)

    Option(typesFile).foreach { file =>
      val typeInfo = names
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
      }.writeTable(output, names.map(_.mkString("\t")))
  }

  def aggregate(keyCond: String, aggCond: String): KeyTable = {

    val aggregationST = fields.zipWithIndex.map { case (fd, i) => (fd.name, (i, fd.`type`)) }.toMap

    val keyEC = EvalContext(aggregationST)
    val ec = EvalContext(fields.zipWithIndex.map { case (fd, i) => (fd.name, (i, TAggregable(fd.`type`, aggregationST))) }.toMap)

    val (keyPaths, keyTypes, keyF) = Parser.parseAnnotationExprs(keyCond, keyEC, None)

    val (aggPaths, aggTypes, aggF) = Parser.parseAnnotationExprs(aggCond, ec, None)

    val keyNames = keyPaths.map(_.head)
    val aggNames = aggPaths.map(_.head)

    val keySignature = TStruct((keyNames, keyTypes).zipped.map { case (n, t) => (n, t) }: _*)
    val valueSignature = TStruct((aggNames, aggTypes).zipped.map { case (n, t) => (n, t) }: _*)

    val localNFields = nFields

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, a) =>
      KeyTable.setEvalContext(ec, a, localNFields)
    })

    val newRDD = KeyTable.toSingleRDD(rdd, nKeys, nValues).mapPartitions { it =>
      it.map { a =>
        KeyTable.setEvalContext(keyEC, a, localNFields)
        val key = Annotation.fromSeq(keyF().map(_.orNull))
        (key, a)
      }
    }.aggregateByKey(zVals)(seqOp, combOp)
      .map { case (k, agg) =>
        resultOp(agg)
        (k, Annotation.fromSeq(aggF().map(_.orNull)))
      }

    KeyTable(newRDD, keySignature, valueSignature)
  }

  def expandTypes(): KeyTable = {
    val localKeySignature = keySignature
    val localValueSignature = valueSignature

    val expandedKeySignature = Annotation.expandType(keySignature).asInstanceOf[TStruct]
    val expandedValueSignature = Annotation.expandType(valueSignature).asInstanceOf[TStruct]

    KeyTable(rdd.map { case (k, v) =>
      (Annotation.expandAnnotation(k, localKeySignature),
        Annotation.expandAnnotation(v, localValueSignature))
    },
      expandedKeySignature,
      expandedValueSignature)
  }

  def flatten(): KeyTable = {
    val localKeySignature = keySignature
    val localValueSignature = valueSignature
    KeyTable(rdd.map { case (k, v) =>
      (Annotation.flattenAnnotation(k, localKeySignature),
        Annotation.flattenAnnotation(v, localValueSignature))
    },
      Annotation.flattenType(keySignature).asInstanceOf[TStruct],
      Annotation.flattenType(valueSignature).asInstanceOf[TStruct])
  }

  def toDF(sqlContext: SQLContext): DataFrame = {
    val localSignature = signature
    sqlContext.createDataFrame(KeyTable.toSingleRDD(rdd, nKeys, nValues)
      .map { a => SparkAnnotationImpex.exportAnnotation(a, localSignature).asInstanceOf[Row] },
      schema.asInstanceOf[StructType])
  }
}

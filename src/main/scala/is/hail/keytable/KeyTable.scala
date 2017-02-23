package is.hail.keytable

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{TStruct, _}
import is.hail.io.exportTypes
import is.hail.methods.{Aggregators, Filter}
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag


object KeyTable {

  def annotationToSeq(a: Annotation, nFields: Int) = Option(a).map(_.asInstanceOf[Row].toSeq).getOrElse(Seq.fill[Any](nFields)(null))

  def setEvalContext(ec: EvalContext, a: Annotation, nFields: Int) =
    ec.setAll(annotationToSeq(a, nFields): _*)

  def fromDF(hc: HailContext, df: DataFrame, keyNames: Array[String]): KeyTable = {
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    KeyTable(hc, df.rdd.map { r =>
      SparkAnnotationImpex.importAnnotation(r, signature)
    },
      signature, keyNames)
  }
}

case class KeyTable(@transient hc: HailContext, @transient rdd: RDD[Annotation],
  signature: TStruct, keyNames: Array[String]) {
  require(fieldNames.areDistinct() && keyNames.areDistinct())
  require(keyNames.forall(fieldNames.contains(_)))

  def fields = signature.fields

  def keyFields = keyNames.flatMap{ n => signature.fieldOption(n)}

  def schema = signature.schema

  def fieldNames = fields.map(_.name).toArray

  def nRows = rdd.count()

  def nFields = fields.length

  def nKeys = keyNames.length

  def typeCheck() = rdd.forall { a => signature.typeCheck(a) }

  def printSchema(): Unit = println(signature.toPrettyString())

  def withKeys(): RDD[(Annotation, Annotation)] = {
    val keyIndices = keyFields.map(_.index)
    val nFieldsLocal = nFields

    rdd.map{ a =>
      val r = KeyTable.annotationToSeq(a, nFieldsLocal).zipWithIndex
      val keyRow = keyIndices.map(i => r(i)._1)
      (Annotation.fromSeq(keyRow), a)
    }
  }

  def same(other: KeyTable): Boolean = {
    if (signature != other.signature) {
      info(
        s"""different signatures:
           | left: ${ signature.toPrettyString() }
           | right: ${ other.signature.toPrettyString() }
           |""".stripMargin)
      false
    } else if (keyNames.toSeq != other.keyNames.toSeq) {
      info(
        s"""different key names:
           | left: ${ keyNames.mkString(", ") }
           | right: ${ other.keyNames.mkString(", ") }
           |""".stripMargin)
      false
    } else {
      val thisFieldNames = fieldNames
      val otherFieldNames = other.fieldNames

      withKeys().groupByKey().fullOuterJoin(other.withKeys().groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (None, None) => true
          case (Some(x), Some(y)) =>
            val r1 = x.map(r => thisFieldNames.zip(r.asInstanceOf[Row].toSeq).toMap).toSet
            val r2 = y.map(r => otherFieldNames.zip(r.asInstanceOf[Row].toSeq).toMap).toSet
            val res = r1 == r2
            if (!res)
              info(s"k=$k r1=${ r1.mkString(",") } r2=${ r2.mkString(",") }")
            res
          case _ =>
            info(s"k=$k v1=$v1 v2=$v2")
            false
        }
      }
    }
  }

  def mapAnnotations[T](f: (Annotation) => T)(implicit tct: ClassTag[T]): RDD[T] = rdd.map(a => f(a))

  def query(code: String): (Type, Querier) = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val nFieldsLocal = nFields

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: (Annotation) => Any = { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)
      f()
    }

    (t, f2)
  }

  def annotate(cond: String): KeyTable = {
    val ec = EvalContext(fields.map(fd => (fd.name, fd.typ)): _*)

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, None)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalSignature = (paths, types).zipped.foldLeft(signature) { case (vs, (ids, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val nFieldsLocal = nFields

    val annotF: Annotation => Annotation = { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)

      f().zip(inserters)
        .foldLeft(a) { case (a1, (v, inserter)) =>
          inserter(a1, v)
        }
    }

    KeyTable(hc, mapAnnotations(annotF), finalSignature, keyNames)
  }

  def filter(p: Annotation => Boolean): KeyTable =
    copy(rdd = rdd.filter { a => p(a) })

  def filter(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val nFieldsLocal = nFields

    val f: () => Boolean = Parser.parseTypedExpr[Boolean](cond, ec)

    val p = (a: Annotation) => {
      KeyTable.setEvalContext(ec, a, nFieldsLocal)
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

    val newSignature = TStruct(fieldTransform.zipWithIndex.map { case (oldIndex, newIndex) => signature.fields(oldIndex).copy(index = newIndex) })
    val nFieldsLocal = nFields

    val selectF: Annotation => Annotation = { a =>
      val row = KeyTable.annotationToSeq(a, nFieldsLocal)
      Annotation.fromSeq(fieldTransform.map { i => row(i) })
    }

    KeyTable(hc, mapAnnotations(selectF), newSignature, newKeys)
  }

  def select(fieldsSelect: java.util.ArrayList[String], newKeys: java.util.ArrayList[String]): KeyTable =
    select(fieldsSelect.asScala.toArray, newKeys.asScala.toArray)

  def rename(fieldNameMap: Map[String, String]): KeyTable = {
    val newSignature = TStruct(signature.fields.map { fd => fd.copy(name = fieldNameMap.getOrElse(fd.name, fd.name)) })
    val newFieldNames = newSignature.fields.map(_.name)
    val newKeyNames = keyNames.map(n => fieldNameMap.getOrElse(n, n))
    val duplicateFieldNames = newFieldNames.foldLeft(Map[String, Int]() withDefaultValue 0) { (m, x) => m + (x -> (m(x) + 1)) }.filter {
      _._2 > 1
    }

    if (duplicateFieldNames.nonEmpty)
      fatal(s"Found duplicate field names after renaming fields: `${ duplicateFieldNames.keys.mkString(", ") }'")

    KeyTable(hc, rdd, newSignature, newKeyNames)
  }

  def rename(newFieldNames: Array[String]): KeyTable = {
    if (newFieldNames.length != nFields)
      fatal(s"Found ${ newFieldNames.length } new field names but need $nFields.")

    rename((fieldNames, newFieldNames).zipped.toMap)
  }

  def rename(fieldNameMap: java.util.HashMap[String, String]): KeyTable = rename(fieldNameMap.asScala.toMap)

  def rename(newFieldNames: java.util.ArrayList[String]): KeyTable = rename(newFieldNames.asScala.toArray)

  def join(other: KeyTable, joinType: String): KeyTable = {
    if (keyNames.length != other.keyNames.length || !(keyFields.map(_.typ) sameElements other.keyFields.map(_.typ)))
      fatal(
        s"""Both KeyTables must have the same number of keys and the types of keys must be identical. Order matters.
           |Left signature: ${ TStruct(keyFields).toPrettyString(compact = true) }
           |Right signature: ${ TStruct(other.keyFields).toPrettyString(compact = true) }""".stripMargin)

    val overlappingFields = fieldNames.toSet.intersect(other.fieldNames.toSet) -- keyNames -- other.keyNames
    if (overlappingFields.nonEmpty)
      fatal(
        s"""Fields that are not keys cannot be present in both KeyTables.
           |Overlapping fields: ${ overlappingFields.mkString(", ") }""".stripMargin)

    val mergeFields = other.fields.filterNot(fd => other.keyNames.contains(fd.name))
    val mergeIndices = mergeFields.map(_.index)

    val newSignature = TStruct((fields ++ mergeFields).map(fd => (fd.name, fd.typ)): _*)

    val size1 = nFields
    val size2 = other.nFields
    val targetSize = newSignature.size

    val merger = (a1: Annotation, a2: Annotation) => {
      if (a1 == null && a2 == null)
        Annotation.empty
      else {
        val s1 = Option(a1).map(_.asInstanceOf[Row].toSeq)
          .getOrElse(Seq.fill[Any](size1)(null))
        val s2 = Option(a2).map(_.asInstanceOf[Row].toSeq)
          .getOrElse(Seq.fill[Any](size2)(null))
        val newValues = s1 ++  mergeIndices.map(i => s2(i))
        assert(newValues.size == targetSize)
        Annotation.fromSeq(newValues)
      }
    }

    val rddLeft = withKeys()
    val rddRight = other.withKeys()

    val joinedRDD = joinType match {
      case "left" => rddLeft.leftOuterJoin(rddRight).map { case (k, (l, r)) => merger(l, r.orNull) }
      case "right" => rddLeft.rightOuterJoin(rddRight).map { case (k, (l, r)) => merger(l.orNull, r) }
      case "inner" => rddLeft.join(rddRight).map { case (k, (l, r)) => merger(l, r) }
      case "outer" => rddLeft.fullOuterJoin(rddRight).map { case (k, (l, r)) => merger(l.orNull, r.orNull) }
      case _ => fatal("Invalid join type specified. Choose one of `left', `right', `inner', `outer'")
    }

    KeyTable(hc, joinedRDD, newSignature, keyNames)
  }

  def forall(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val nFieldsLocal = nFields

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    rdd.forall { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def exists(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val nFieldsLocal = nFields

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    rdd.exists { a =>
      KeyTable.setEvalContext(ec, a, nFieldsLocal)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def export(sc: SparkContext, output: String, typesFile: String) {
    val hConf = sc.hadoopConfiguration

    val ec = EvalContext(fields.map(fd => (fd.name, fd.typ)): _*)

    Option(typesFile).foreach { file =>
      exportTypes(file, hConf, fields.map(f => (f.name, f.typ)).toArray)
    }

    hConf.delete(output, recursive = true)

    val localNFields = nFields
    val localTypes = fields.map(_.typ)

    rdd.mapPartitions { it =>
        val sb = new StringBuilder()
        val nulls: Seq[Any] = Array.fill[Annotation](localNFields)(null)

        it.map { r =>
          sb.clear()

          val r2 =
            if (r == null)
              nulls
            else
              r.asInstanceOf[Row].toSeq
          assert(r2.length == localTypes.length)

          r2.zip(localTypes)
            .foreachBetween { case (x, t) =>
              sb.append(t.str(x))
            }(sb += '\t')

          sb.result()
        }
      }.writeTable(output, hc.tmpDir, Some(fields.map(_.name).mkString("\t")))
  }

  def aggregate(keyCond: String, aggCond: String): KeyTable = {

    val aggregationST = fields.zipWithIndex.map {
      case (fd, i) => (fd.name, (i, fd.typ))
    }.toMap

    val keyEC = EvalContext(aggregationST)
    val ec = EvalContext(fields.zipWithIndex.map {
      case (fd, i) => (fd.name, (i, TAggregable(fd.typ, aggregationST)))
    }.toMap)

    val (keyPaths, keyTypes, keyF) = Parser.parseAnnotationExprs(keyCond, keyEC, None)

    val (aggPaths, aggTypes, aggF) = Parser.parseAnnotationExprs(aggCond, ec, None)

    val newKeyNames = keyPaths.map(_.head)
    val aggNames = aggPaths.map(_.head)

    val keySignature = TStruct((newKeyNames, keyTypes).zipped.map {
      case (n, t) => (n, t)
    }: _*)
    val aggSignature = TStruct((aggNames, aggTypes).zipped.map {
      case (n, t) => (n, t)
    }: _*)

    val nFieldsLocal = nFields
    val nKeysLocal = nKeys

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Annotation](ec, {
      case (ec, a) =>
        KeyTable.setEvalContext(ec, a, nFieldsLocal)
    })

    val newRDD = rdd.mapPartitions {
      it =>
        it.map {
          a =>
            KeyTable.setEvalContext(keyEC, a, nFieldsLocal)
            val key = Annotation.fromSeq(keyF())
            (key, a)
        }
    }.aggregateByKey(zVals)(seqOp, combOp)
      .map {
        case (k, agg) =>
          resultOp(agg)
          Annotation.fromSeq(KeyTable.annotationToSeq(k, nKeysLocal) ++ aggF())
      }

    KeyTable(hc, newRDD, keySignature.merge(aggSignature)._1, newKeyNames)
  }

  def expandTypes(): KeyTable = {
    val localSignature = signature
    val expandedSignature = Annotation.expandType(localSignature).asInstanceOf[TStruct]

    KeyTable(hc, rdd.map { a => Annotation.expandAnnotation(a, localSignature) },
      expandedSignature,
      keyNames)
  }

  def flatten(): KeyTable = {
    val localSignature = signature
    KeyTable(hc, rdd.map { a => Annotation.flattenAnnotation(a, localSignature) },
      Annotation.flattenType(signature).asInstanceOf[TStruct],
      keyNames)
  }

  def toDF(sqlContext: SQLContext): DataFrame = {
    val localSignature = signature
    sqlContext.createDataFrame(
      rdd.map {
        a => SparkAnnotationImpex.exportAnnotation(a, localSignature).asInstanceOf[Row]
      },
      schema.asInstanceOf[StructType])
  }

  def explode(columnName: String): KeyTable = {

    val explodeField = signature.fieldOption(columnName) match {
      case Some(x) => x
      case None =>
        fatal(
          s"""Input field name `${ columnName }' not found in KeyTable.
             |KeyTable field names are `${ fieldNames.mkString(", ") }'.""".stripMargin)
    }

    val index = explodeField.index

    val explodeType = explodeField.typ match {
      case t: TIterable => t.elementType
      case _ => fatal(s"Require Array or Set. Column `$columnName' has type `${ explodeField.typ }'.")
    }

    val newSignature = signature.copy(fields = fields.updated(index, Field(columnName, explodeType, index)))

    val nFieldsLocal = nFields

    val explodedRDD = rdd.flatMap { a =>
      val row = KeyTable.annotationToSeq(a, nFieldsLocal)
      for (element <- row(index).asInstanceOf[Iterable[_]]) yield row.updated(index, element)
    }.map(Annotation.fromSeq)

    KeyTable(hc, explodedRDD, newSignature, keyNames)
  }

  def explode(columnNames: Array[String]): KeyTable = {
    columnNames.foldLeft(this)((kt, name) => kt.explode(name))
  }

  def explode(columnNames: java.util.ArrayList[String]): KeyTable = explode(columnNames.asScala.toArray)
}

package is.hail.expr

import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils.{Interval, _}
import is.hail.variant._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

abstract class AnnotationImpex[T, A] {
  // FIXME for now, schema must be specified on import
  def exportType(t: Type): T

  def exportAnnotation(a: Annotation, t: Type): A

  def importAnnotation(a: A, t: Type): Annotation
}

object SparkAnnotationImpex extends AnnotationImpex[DataType, Any] {
  val invalidCharacters: Set[Char] = " ,;{}()\n\t=".toSet

  def escapeColumnName(name: String): String = {
    name.map { c =>
      if (invalidCharacters.contains(c))
        "_"
      else
        c
    }.mkString
  }

  def requiresConversion(t: Type): Boolean = t match {
    case TArray(elementType, _) => requiresConversion(elementType)
    case TSet(_, _) | TDict(_, _, _) | TAltAllele(_) | TVariant(_, _) | TLocus(_, _) | TInterval(_, _) | TCall(_) => true
    case TStruct(fields, _) =>
      fields.isEmpty || fields.exists(f => requiresConversion(f.typ))
    case _ => false
  }

  def importType(t: DataType): Type = t match {
    case BooleanType => TBoolean()
    case IntegerType => TInt32()
    case LongType => TInt64()
    case FloatType => TFloat32()
    case DoubleType => TFloat64()
    case StringType => TString()
    case BinaryType => TBinary()
    case ArrayType(elementType, containsNull) => TArray(importType(elementType).setRequired(!containsNull))
    case StructType(fields) =>
      TStruct(fields.zipWithIndex
        .map { case (f, i) =>
          (f.name, importType(f.dataType).setRequired(!f.nullable))
        }: _*)
  }

  def annotationImporter(t: Type): (Any) => Annotation = {
    if (requiresConversion(t))
      (a: Any) => importAnnotation(a, t)
    else
      (a: Any) => a
  }

  def importAnnotation(a: Any, t: Type): Annotation = {
    if (a == null) {
      if (t.required)
        fatal("required annotation cannot be null")
      null
    } else
      t match {
        case TArray(elementType, _) =>
          a.asInstanceOf[Seq[_]].map(elem => importAnnotation(elem, elementType))
        case TSet(elementType, _) =>
          a.asInstanceOf[Seq[_]].map(elem => importAnnotation(elem, elementType)).toSet
        case TDict(keyType, valueType, _) =>
          val kvPairs = a.asInstanceOf[IndexedSeq[Annotation]]
          kvPairs
            .map(_.asInstanceOf[Row])
            .map(r => (importAnnotation(r.get(0), keyType), importAnnotation(r.get(1), valueType)))
            .toMap
        case TCall(_) =>
          Call.parse(a.asInstanceOf[String])
        case TAltAllele(_) =>
          val r = a.asInstanceOf[Row]
          AltAllele(r.getAs[String](0), r.getAs[String](1))
        case _: TVariant =>
          val r = a.asInstanceOf[Row]
          Variant(r.getAs[String](0), r.getAs[Int](1), r.getAs[String](2),
            r.getAs[Seq[Row]](3).map(aa =>
              importAnnotation(aa, TAltAllele()).asInstanceOf[AltAllele]).toArray)
        case _: TLocus =>
          val r = a.asInstanceOf[Row]
          Locus(r.getAs[String](0), r.getAs[Int](1))
        case x: TInterval =>
          val r = a.asInstanceOf[Row]
          Interval(importAnnotation(r.get(0), x.pointType),
            importAnnotation(r.get(1), x.pointType),
            importAnnotation(r.get(2), TBooleanRequired).asInstanceOf[Boolean],
            importAnnotation(r.get(3), TBooleanRequired).asInstanceOf[Boolean])
        case TStruct(fields, _) =>
          if (fields.isEmpty)
            if (a.asInstanceOf[Boolean]) Annotation.empty else null
          else {
            val r = a.asInstanceOf[Row]
            Annotation.fromSeq(r.toSeq.zip(fields).map { case (v, f) =>
              importAnnotation(v, f.typ)
            })
          }
        case _ => a
      }
  }

  def exportType(t: Type): DataType = (t: @unchecked) match {
    case _: TBoolean => BooleanType
    case _: TInt32 => IntegerType
    case _: TInt64 => LongType
    case _: TFloat32 => FloatType
    case _: TFloat64 => DoubleType
    case _: TString => StringType
    case _: TBinary => BinaryType
    case TArray(elementType, _) =>
      ArrayType(exportType(elementType), containsNull = !elementType.required)
    case TSet(elementType, req) =>
      ArrayType(exportType(elementType), containsNull = !elementType.required)
    case TDict(keyType, valueType, _) =>
      ArrayType(StructType(Array(
        StructField("key", keyType.schema, nullable = !keyType.required),
        StructField("value", valueType.schema, nullable = !valueType.required))), containsNull = false)
    case _: TAltAllele => AltAllele.sparkSchema
    case _: TVariant => Variant.sparkSchema
    case _: TLocus => Locus.sparkSchema
    case _: TInterval => StructType(Array(
      StructField("start", Locus.sparkSchema, nullable = false),
      StructField("end", Locus.sparkSchema, nullable = false)))
    case _: TCall => StringType
    case TStruct(fields, _) =>
      if (fields.isEmpty)
        BooleanType //placeholder
      else
        StructType(fields
          .map(f =>
            StructField(escapeColumnName(f.name), f.typ.schema, nullable = !f.typ.required)))
  }

  def annotationExporter(t: Type): (Annotation) => Any = {
    if (requiresConversion(t))
      (a: Annotation) => exportAnnotation(a, t)
    else
      (a: Annotation) => a
  }

  def exportAnnotation(a: Annotation, t: Type): Any = {
    if (a == null) {
      if (t.required)
        fatal("required annotation cannot be null")
      null
    } else
      t match {
        case TArray(elementType, _) =>
          a.asInstanceOf[IndexedSeq[_]].map(elem => exportAnnotation(elem, elementType))
        case TSet(elementType, _) =>
          a.asInstanceOf[Set[_]].toSeq.map(elem => exportAnnotation(elem, elementType))
        case TDict(keyType, valueType, _) =>
          a.asInstanceOf[Map[_, _]]
            .map { case (k, v) =>
              Row.fromSeq(Seq(exportAnnotation(k, keyType), exportAnnotation(v, valueType)))
            }.toIndexedSeq
        case TCall(_) =>
          Call.toString(a.asInstanceOf[Call])
        case TAltAllele(_) =>
          val aa = a.asInstanceOf[AltAllele]
          Row(aa.ref, aa.alt)
        case TVariant(gr, _) =>
          val v = a.asInstanceOf[Variant]
          Row(v.contig, v.start, v.ref, v.altAlleles.map(aa => Row(aa.ref, aa.alt)))
        case TLocus(gr, _) =>
          val l = a.asInstanceOf[Locus]
          Row(l.contig, l.position)
        case TInterval(pointType, _) =>
          val i = a.asInstanceOf[Interval]
          Row(exportAnnotation(i.start, pointType),
            exportAnnotation(i.end, pointType),
            exportAnnotation(i.includeStart, TBooleanRequired),
            exportAnnotation(i.includeEnd, TBooleanRequired))
        case TStruct(fields, _) =>
          if (fields.isEmpty)
            a != null
          else {
            val r = a.asInstanceOf[Row]
            Annotation.fromSeq(r.toSeq.zip(fields).map {
              case (v, f) => exportAnnotation(v, f.typ)
            })
          }
        case _ => a
      }
  }
}

case class JSONExtractVariant(contig: String,
  start: Int,
  ref: String,
  altAlleles: List[AltAllele]) {
  def toVariant =
    Variant(contig, start, ref, altAlleles.toArray)
}

case class JSONExtractIntervalLocus(start: Locus, end: Locus) {
  def toLocusTuple: (Locus, Locus) = (start, end)
}

case class JSONExtractContig(name: String, length: Int)

case class JSONExtractGenomeReference(name: String, contigs: Array[JSONExtractContig], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[JSONExtractIntervalLocus]) {

  def toGenomeReference: GenomeReference = GenomeReference(name, contigs.map(_.name),
    contigs.map(c => (c.name, c.length)).toMap, xContigs, yContigs, mtContigs, par.map(_.toLocusTuple))
}

object JSONAnnotationImpex extends AnnotationImpex[Type, JValue] {
  def jsonExtractVariant(t: Type, variantFields: String): Any => Variant = {
    val ec = EvalContext(Map(
      "root" -> (0, t)))

    val (types, f) = Parser.parseExprs(variantFields, ec)

    if (types.length != 4)
      fatal(s"wrong number of variant field expressions: expected 4, got ${ types.length }")

    if (!types(0).isInstanceOf[TString])
      fatal(s"wrong type for chromosome field: expected String, got ${ types(0) }")
    if (types(1).isInstanceOf[TInt32])
      fatal(s"wrong type for pos field: expected Int, got ${ types(1) }")
    if (types(2).isInstanceOf[TString])
      fatal(s"wrong type for ref field: expected String, got ${ types(2) }")
    if (types(3) != TArray(TString()) && types(3) != TArray(+TString()))
      fatal(s"wrong type for alt field: expected Array[String], got ${ types(3) }")

    (root: Annotation) => {
      ec.setAll(root)

      val vfs = f()

      val chr = vfs(0)
      val pos = vfs(1)
      val ref = vfs(2)
      val alt = vfs(3)

      if (chr != null && pos != null && ref != null && alt != null)
        Variant(chr.asInstanceOf[String],
          pos.asInstanceOf[Int],
          ref.asInstanceOf[String],
          alt.asInstanceOf[IndexedSeq[String]].toArray)
      else
        null
    }
  }

  def jsonExtractSample(t: Type, sampleExpr: String): Any => String = {
    val ec = EvalContext(Map(
      "root" -> (0, t)))

    val f: () => String = Parser.parseTypedExpr[String](sampleExpr, ec)

    (root: Annotation) => {
      ec.setAll(root)
      f()
    }
  }

  def exportType(t: Type): Type = t

  def exportAnnotation(a: Annotation, t: Type): JValue =
    if (a == null)
      JNull
    else {
      (t: @unchecked) match {
        case _: TBoolean => JBool(a.asInstanceOf[Boolean])
        case _: TInt32 => JInt(a.asInstanceOf[Int])
        case _: TInt64 => JInt(a.asInstanceOf[Long])
        case _: TFloat32 => JDouble(a.asInstanceOf[Float])
        case _: TFloat64 => JDouble(a.asInstanceOf[Double])
        case _: TString => JString(a.asInstanceOf[String])
        case TArray(elementType, _) =>
          val arr = a.asInstanceOf[Seq[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType)).toList)
        case TSet(elementType, _) =>
          val arr = a.asInstanceOf[Set[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType)).toList)
        case TDict(keyType, valueType, _) =>
          val m = a.asInstanceOf[Map[_, _]]
          JArray(m.map { case (k, v) => JObject(
            "key" -> exportAnnotation(k, keyType),
            "value" -> exportAnnotation(v, valueType))
          }.toList)
        case _: TCall => JString(Call.toString(a.asInstanceOf[Call]))
        case _: TAltAllele => a.asInstanceOf[AltAllele].toJSON
        case TVariant(_, _) => a.asInstanceOf[Variant].toJSON
        case TLocus(_, _) => a.asInstanceOf[Locus].toJSON
        case TInterval(pointType, _) => a.asInstanceOf[Interval].toJSON(pointType.toJSON)
        case TStruct(fields, _) =>
          val row = a.asInstanceOf[Row]
          JObject(fields
            .map(f => (f.name, exportAnnotation(row.get(f.index), f.typ)))
            .toList)
      }
    }

  def importAnnotation(jv: JValue, t: Type): Annotation =
    importAnnotation(jv, t, "<root>")

  def importAnnotation(jv: JValue, t: Type, parent: String): Annotation = {
    implicit val formats = Serialization.formats(NoTypeHints)

    (jv, t) match {
      case (JNull | JNothing, _) =>
        if (t.required)
          fatal("required annotation cannot be null")
        null
      case (JInt(x), _: TInt32) => x.toInt
      case (JInt(x), _: TInt64) => x.toLong
      case (JInt(x), _: TFloat64) => x.toDouble
      case (JInt(x), _: TString) => x.toString
      case (JDouble(x), _: TFloat64) => x
      case (JString("Infinity"), _: TFloat64) => Double.PositiveInfinity
      case (JString("-Infinity"), _: TFloat64) => Double.NegativeInfinity
      case (JString("Infinity"), _: TFloat32) => Float.PositiveInfinity
      case (JString("-Infinity"), _: TFloat32) => Float.NegativeInfinity
      case (JDouble(x), _: TFloat32) => x.toFloat
      case (JString(x), _: TString) => x
      case (JString(x), _: TInt32) =>
        x.toInt
      case (JString(x), _: TFloat64) =>
        if (x.startsWith("-:"))
          x.drop(2).toDouble
        else
          x.toDouble
      case (JBool(x), _: TBoolean) => x

      // back compatibility
      case (JObject(a), TDict(TString(_), valueType, _)) =>
        a.map { case (key, value) =>
          (key, importAnnotation(value, valueType, parent))
        }
          .toMap

      case (JArray(arr), TDict(keyType, valueType, _)) =>
        arr.map { case JObject(a) =>
          a match {
            case List(k, v) =>
              (k, v) match {
                case (("key", ka), ("value", va)) =>
                  (importAnnotation(ka, keyType, parent), importAnnotation(va, valueType, parent))
              }
            case _ =>
              warn(s"Can't convert JSON value $jv to type $t at $parent.")
              null

          }
        case _ =>
          warn(s"Can't convert JSON value $jv to type $t at $parent.")
          null
        }.toMap

      case (JObject(jfields), t: TStruct) =>
        if (t.size == 0)
          Annotation.empty
        else {
          val a = Array.fill[Any](t.size)(null)

          for ((name, jv2) <- jfields) {
            t.selfField(name) match {
              case Some(f) =>
                a(f.index) = importAnnotation(jv2, f.typ, parent + "." + name)

              case None =>
                warn(s"$t has no field $name at $parent")
            }
          }

          Annotation(a: _*)
        }
      case (_, _: TAltAllele) =>
        jv.extract[AltAllele]
      case (_, TVariant(_, _)) =>
        jv.extract[JSONExtractVariant].toVariant
      case (_, TLocus(_, _)) =>
        jv.extract[Locus]
      case (_, TInterval(pointType, _)) =>
        jv match {
          case JObject(List(("start", sjv), ("end", ejv), ("includeStart", isjv), ("includeEnd", iejv))) =>
            Interval(importAnnotation(sjv, pointType, parent + ".start"),
              importAnnotation(ejv, pointType, parent + ".end"),
              importAnnotation(isjv, TBooleanRequired, parent + ".includeStart").asInstanceOf[Boolean],
              importAnnotation(iejv, TBooleanRequired, parent + ".includeEnd").asInstanceOf[Boolean]
            )
          case _ =>
            warn(s"Can't convert JSON value $jv to type $t at $parent.")
            null
        }
      case (JString(x), _: TCall) => Call.parse(x)

      case (JArray(a), TArray(elementType, _)) =>
        a.iterator.map(jv2 => importAnnotation(jv2, elementType, parent + ".<array>")).toArray[Any]: IndexedSeq[Any]

      case (JArray(a), TSet(elementType, _)) =>
        a.iterator.map(jv2 => importAnnotation(jv2, elementType, parent + ".<array>")).toSet[Any]

      case _ =>
        warn(s"Can't convert JSON value $jv to type $t at $parent.")
        null
    }
  }
}

object TableAnnotationImpex extends AnnotationImpex[Unit, String] {

  // Tables have no schema
  def exportType(t: Type): Unit = ()

  def exportAnnotation(a: Annotation, t: Type): String = {
    if (a == null)
      "NA"
    else {
      t match {
        case _: TFloat64 => a.asInstanceOf[Double].formatted("%.4e")
        case _: TString => a.asInstanceOf[String]
        case d: TDict => JsonMethods.compact(d.toJSON(a))
        case it: TIterable => JsonMethods.compact(it.toJSON(a))
        case t: TStruct => JsonMethods.compact(t.toJSON(a))
        case TInterval(TLocus(gr, _), _) =>
          val i = a.asInstanceOf[Interval]
          val bounds = if (i.start.asInstanceOf[Locus].contig == i.end.asInstanceOf[Locus].contig)
            s"${ i.start }-${ i.end.asInstanceOf[Locus].position }"
          else
            s"${ i.start }-${ i.end }"
          if (!i.includeStart || i.includeEnd)
            s"${if (i.includeStart) "[" else "("}$bounds${if (i.includeEnd) "]" else ")"}"
          else
            bounds
        case _: TInterval =>
          JsonMethods.compact(t.toJSON(a))
        case _: TCall => Call.toString(a.asInstanceOf[Call])
        case _ => a.toString
      }
    }
  }

  def importAnnotation(a: String, t: Type): Annotation = {
    (t: @unchecked) match {
      case _: TString => a
      case _: TInt32 => a.toInt
      case _: TInt64 => a.toLong
      case _: TFloat32 => a.toFloat
      case _: TFloat64 => if (a == "nan") Double.NaN else a.toDouble
      case _: TBoolean => a.toBoolean
      case tl: TLocus => Locus.parse(a, tl.gr)
        // FIXME legacy
      case TInterval(TLocus(gr, _), _) => Locus.parseInterval(a, gr)
      case t: TInterval => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case tv: TVariant => Variant.parse(a, tv.gr)
      case _: TAltAllele => a.split("/") match {
        case Array(ref, alt) => AltAllele(ref, alt)
      }
      case _: TCall => Call.parse(a)
      case t: TArray => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TSet => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TDict => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TStruct => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
    }
  }
}

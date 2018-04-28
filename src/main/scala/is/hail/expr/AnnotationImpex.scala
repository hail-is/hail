package is.hail.expr

import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils.{Interval, _}
import is.hail.variant._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

object SparkAnnotationImpex {
  val invalidCharacters: Set[Char] = " ,;{}()\n\t=".toSet

  def escapeColumnName(name: String): String = {
    name.map { c =>
      if (invalidCharacters.contains(c))
        "_"
      else
        c
    }.mkString
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
      TStruct(fields.map { f => (f.name, importType(f.dataType).setRequired(!f.nullable)) }: _*)
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
    case tbs: TStruct =>
      if (tbs.fields.isEmpty)
        BooleanType //placeholder
      else
        StructType(tbs.fields
          .map(f =>
            StructField(escapeColumnName(f.name), f.typ.schema, nullable = !f.typ.required)))
  }
}

case class JSONExtractIntervalLocus(start: Locus, end: Locus) {
  def toLocusTuple: (Locus, Locus) = (start, end)
}

case class JSONExtractContig(name: String, length: Int)

case class JSONExtractReferenceGenome(name: String, contigs: Array[JSONExtractContig], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[JSONExtractIntervalLocus]) {

  def toReferenceGenome: ReferenceGenome = ReferenceGenome(name, contigs.map(_.name),
    contigs.map(c => (c.name, c.length)).toMap, xContigs, yContigs, mtContigs, par.map(_.toLocusTuple))
}

object JSONAnnotationImpex {
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
        case TLocus(_, _) => a.asInstanceOf[Locus].toJSON
        case TInterval(pointType, _) => a.asInstanceOf[Interval].toJSON(pointType.toJSON)
        case TStruct(fields, _) =>
          val row = a.asInstanceOf[Row]
          JObject(fields
            .map(f => (f.name, exportAnnotation(row.get(f.index), f.typ)))
            .toList)
        case TTuple(types, _) =>
          val row = a.asInstanceOf[Row]
          JArray(types.zipWithIndex.map { case (typ, i) => exportAnnotation(row.get(i), typ) }.toList)
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
      case (JArray(elts), t: TTuple) =>
        if (t.size == 0)
          Annotation.empty
        else {
          val a = Array.fill[Any](t.size)(null)
          var i = 0
          for (jvelt <- elts) {
            a(i) = importAnnotation(jvelt, t.types(i), parent)
            i += 1
          }

          Annotation(a: _*)
        }
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

object TableAnnotationImpex {

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
        case t: TBaseStruct => JsonMethods.compact(t.toJSON(a))
        case TInterval(TLocus(rg, _), _) =>
          val i = a.asInstanceOf[Interval]
          val bounds = if (i.start.asInstanceOf[Locus].contig == i.end.asInstanceOf[Locus].contig)
            s"${ i.start }-${ i.end.asInstanceOf[Locus].position }"
          else
            s"${ i.start }-${ i.end }"
          s"${ if (i.includesStart) "[" else "(" }$bounds${ if (i.includesEnd) "]" else ")" }"
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
      case tl: TLocus => Locus.parse(a, tl.rg)
      // FIXME legacy
      case TInterval(TLocus(rg, _), _) => Locus.parseInterval(a, rg)
      case t: TInterval => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case _: TCall => Call.parse(a)
      case t: TArray => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TSet => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TDict => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TBaseStruct => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
    }
  }
}

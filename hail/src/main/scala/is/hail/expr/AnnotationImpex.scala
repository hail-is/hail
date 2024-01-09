package is.hail.expr

import is.hail.annotations.{Annotation, NDArray, SafeNDArray, UnsafeNDArray}
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.types.physical.{
  PBoolean, PCanonicalArray, PCanonicalBinary, PCanonicalString, PCanonicalStruct, PFloat32,
  PFloat64, PInt32, PInt64, PType,
}
import is.hail.types.virtual._
import is.hail.utils.{Interval, _}
import is.hail.variant._

import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.mutable

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s

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

  def importType(t: DataType): PType = t match {
    case BooleanType => PBoolean()
    case IntegerType => PInt32()
    case LongType => PInt64()
    case FloatType => PFloat32()
    case DoubleType => PFloat64()
    case StringType => PCanonicalString()
    case BinaryType => PCanonicalBinary()
    case ArrayType(elementType, containsNull) =>
      PCanonicalArray(importType(elementType).setRequired(!containsNull))
    case StructType(fields) =>
      PCanonicalStruct(fields.map { f =>
        (f.name, importType(f.dataType).setRequired(!f.nullable))
      }: _*)
  }

  def exportType(t: Type): DataType = (t: @unchecked) match {
    case TBoolean => BooleanType
    case TInt32 => IntegerType
    case TInt64 => LongType
    case TFloat32 => FloatType
    case TFloat64 => DoubleType
    case TString => StringType
    case TBinary => BinaryType
    case TArray(elementType) =>
      ArrayType(exportType(elementType), containsNull = true)
    case tbs: TBaseStruct =>
      if (tbs.fields.isEmpty)
        BooleanType // placeholder
      else
        StructType(tbs.fields
          .map(f =>
            StructField(escapeColumnName(f.name), f.typ.schema, nullable = true)
          ))
  }
}

case class JSONExtractIntervalLocus(start: Locus, end: Locus) {
  def toLocusTuple: (Locus, Locus) = (start, end)
}

case class JSONExtractContig(name: String, length: Int)

case class JSONExtractReferenceGenome(
  name: String,
  contigs: Array[JSONExtractContig],
  xContigs: Set[String],
  yContigs: Set[String],
  mtContigs: Set[String],
  par: Array[JSONExtractIntervalLocus],
) {

  def toReferenceGenome: ReferenceGenome = ReferenceGenome(
    name,
    contigs.map(_.name),
    contigs.map(c => (c.name, c.length)).toMap,
    xContigs,
    yContigs,
    mtContigs,
    par.map(_.toLocusTuple),
  )
}

object JSONAnnotationImpex {
  implicit val serializationFormats: json4s.Formats = Serialization.formats(NoTypeHints)

  def exportType(t: Type): Type = t

  val doubleConv = Map(
    "nan" -> Double.NaN,
    "NaN" -> Double.NaN,
    "inf" -> Double.PositiveInfinity,
    "Infinity" -> Double.PositiveInfinity,
    "-inf" -> Double.NegativeInfinity,
    "-Infinity" -> Double.NegativeInfinity,
  )

  val floatConv = Map(
    "nan" -> Float.NaN,
    "NaN" -> Float.NaN,
    "inf" -> Float.PositiveInfinity,
    "Infinity" -> Float.PositiveInfinity,
    "-inf" -> Float.NegativeInfinity,
    "-Infinity" -> Float.NegativeInfinity,
  )

  def exportAnnotation(a: Annotation, t: Type): JValue =
    try
      _exportAnnotation(a, t)
    catch {
      case exc: Exception =>
        fatal(s"Could not export annotation with type $t: $a", exc)
    }

  def _exportAnnotation(a: Annotation, t: Type): JValue =
    if (a == null)
      JNull
    else {
      (t: @unchecked) match {
        case TBoolean => JBool(a.asInstanceOf[Boolean])
        case TInt32 => JInt(a.asInstanceOf[Int])
        case TInt64 => JInt(a.asInstanceOf[Long])
        case TFloat32 => JDouble(a.asInstanceOf[Float])
        case TFloat64 => JDouble(a.asInstanceOf[Double])
        case TString => JString(a.asInstanceOf[String])
        case TVoid =>
          JNull
        case TArray(elementType) =>
          val arr = a.asInstanceOf[Seq[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType)).toList)
        case TSet(elementType) =>
          val arr = a.asInstanceOf[Set[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType)).toList)
        case TDict(keyType, valueType) =>
          val m = a.asInstanceOf[Map[_, _]]
          JArray(m.map { case (k, v) =>
            JObject(
              "key" -> exportAnnotation(k, keyType),
              "value" -> exportAnnotation(v, valueType),
            )
          }.toList)
        case TCall => JString(Call.toString(a.asInstanceOf[Call]))
        case TLocus(_) => a.asInstanceOf[Locus].toJSON
        case TInterval(pointType) => a.asInstanceOf[Interval].toJSON(pointType.toJSON)
        case TStruct(fields) =>
          val row = a.asInstanceOf[Row]
          JObject(List.tabulate(row.size) { i =>
            (fields(i).name, exportAnnotation(row.get(i), fields(i).typ))
          })
        case TTuple(types) =>
          val row = a.asInstanceOf[Row]
          JArray(List.tabulate(row.size)(i => exportAnnotation(row.get(i), types(i).typ)))
        case TNDArray(elementType, _) =>
          val jnd = a.asInstanceOf[NDArray]
          JObject(
            "shape" -> JArray(jnd.shape.map(shapeEntry => JInt(shapeEntry)).toList),
            "data" -> JArray(jnd.getRowMajorElements().map(a =>
              exportAnnotation(a, elementType)
            ).toList),
          )
      }
    }

  def irImportAnnotation(s: String, t: Type, warnContext: mutable.HashSet[String]): Row = {
    try
      // wraps in a Row to handle returned missingness
      Row(importAnnotation(JsonMethods.parse(s), t, true, warnContext))
    catch {
      case e: Throwable =>
        fatal(s"Error parsing JSON:\n  type: $t\n  value: $s", e)
    }
  }

  def importAnnotation(
    jv: JValue,
    t: Type,
    padNulls: Boolean = true,
    warnContext: mutable.HashSet[String] = null,
  ): Annotation =
    importAnnotationInternal(
      jv,
      t,
      "<root>",
      padNulls,
      if (warnContext == null) new mutable.HashSet[String] else warnContext,
    )

  private def importAnnotationInternal(
    jv: JValue,
    t: Type,
    parent: String,
    padNulls: Boolean,
    warnContext: mutable.HashSet[String],
  ): Annotation = {
    def imp(jv: JValue, t: Type, parent: String): Annotation =
      importAnnotationInternal(jv, t, parent, padNulls, warnContext)
    def warnOnce(msg: String, path: String): Unit =
      if (!warnContext.contains(path)) {
        warn(msg)
        warnContext += path
      }

    (jv, t) match {
      case (JNull | JNothing, _) => null
      case (JInt(x), TInt32) => x.toInt
      case (JInt(x), TInt64) => x.toLong
      case (JInt(x), TFloat64) => x.toDouble
      case (JInt(x), TString) => x.toString
      case (JDouble(x), TFloat64) => x
      case (JDouble(x), TFloat32) => x.toFloat
      case (JString(x), TFloat64) if doubleConv.contains(x) => doubleConv(x)
      case (JString(x), TFloat32) if floatConv.contains(x) => floatConv(x)
      case (JString(x), TString) => x
      case (JString(x), TInt32) =>
        x.toInt
      case (JString(x), TFloat64) =>
        if (x.startsWith("-:"))
          x.drop(2).toDouble
        else
          x.toDouble
      case (JBool(x), TBoolean) => x

      // back compatibility
      case (JObject(a), TDict(TString, valueType)) =>
        a.map { case (key, value) =>
          (key, imp(value, valueType, parent))
        }
          .toMap

      case (JArray(arr), TDict(keyType, valueType)) =>
        val keyPath = parent + "[key]"
        val valuePath = parent + "[value]"
        arr.map {
          case JObject(a) =>
            a match {
              case List(k, v) =>
                (k, v) match {
                  case (("key", ka), ("value", va)) =>
                    (imp(ka, keyType, keyPath), imp(va, valueType, valuePath))
                }
              case _ =>
                warnOnce(s"Can't convert JSON value $jv to type $t at $parent.", parent)
                null

            }
          case _ =>
            warnOnce(s"Can't convert JSON value $jv to type $t at $parent.", parent)
            null
        }.toMap

      case (JObject(jfields), t: TStruct) =>
        if (t.size == 0)
          Annotation.empty
        else {
          val annotationSize =
            if (padNulls) t.size
            else jfields.map { case (name, jv2) =>
              t.selfField(name).map(_.index).getOrElse(-1)
            }.max + 1
          val a = Array.fill[Any](annotationSize)(null)

          for ((name, jv2) <- jfields) {
            t.selfField(name) match {
              case Some(f) =>
                a(f.index) = imp(jv2, f.typ, parent + "." + name)

              case None =>
                warnOnce(s"$t has no field $name at $parent for value $jv2", parent + "/" + name)
            }
          }

          Annotation.fromSeq(a)
        }
      case (JArray(elts), t: TTuple) =>
        if (t.size == 0)
          Annotation.empty
        else {
          val annotationSize =
            if (padNulls) t.size
            else elts.length
          val a = Array.fill[Any](annotationSize)(null)
          var i = 0
          for (jvelt <- elts) {
            a(i) = imp(jvelt, t.types(i), parent)
            i += 1
          }

          Annotation.fromSeq(a)
        }
      case (_, TLocus(_)) =>
        jv.extract[Locus]
      case (
            JObject(List(("shape", shapeJson: JArray), ("data", dataJson: JArray))),
            t @ TNDArray(_, _),
          ) =>
        val shapeArray =
          shapeJson.arr.map(imp(_, TInt64, parent)).map(_.asInstanceOf[Long]).toIndexedSeq
        val dataArray = dataJson.arr.map(imp(_, t.elementType, parent)).toIndexedSeq

        new SafeNDArray(shapeArray, dataArray)
      case (_, TInterval(pointType)) =>
        jv match {
          case JObject(list) =>
            val m = list.toMap
            (m.get("start"), m.get("end"), m.get("includeStart"), m.get("includeEnd")) match {
              case (Some(sjv), Some(ejv), Some(isjv), Some(iejv)) =>
                Interval(
                  imp(sjv, pointType, parent + ".start"),
                  imp(ejv, pointType, parent + ".end"),
                  imp(isjv, TBoolean, parent + ".includeStart").asInstanceOf[Boolean],
                  imp(iejv, TBoolean, parent + ".includeEnd").asInstanceOf[Boolean],
                )
              case _ =>
                warnOnce(s"Can't convert JSON value $jv to type $t at $parent.", parent)
                null
            }
          case _ =>
            warnOnce(s"Can't convert JSON value $jv to type $t at $parent.", parent)
            null
        }
      case (JString(x), TCall) => Call.parse(x)

      case (JArray(a), TArray(elementType)) =>
        a.iterator.map(jv2 => imp(jv2, elementType, parent + "[element]")).toArray[Any]: IndexedSeq[
          Any
        ]

      case (JArray(a), TSet(elementType)) =>
        a.iterator.map(jv2 => imp(jv2, elementType, parent + "[element]")).toSet[Any]
      case _ =>
        warnOnce(s"Can't convert JSON value $jv to type $t at $parent.", parent)
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
        case TFloat64 => "%.4e".format(a.asInstanceOf[Double])
        case TString => a.asInstanceOf[String]
        case t: TContainer => JsonMethods.compact(t.toJSON(a))
        case t: TBaseStruct => JsonMethods.compact(t.toJSON(a))
        case t: TNDArray => JsonMethods.compact(t.toJSON(a))
        case TInterval(TLocus(_)) =>
          val i = a.asInstanceOf[Interval]
          val bounds = if (i.start.asInstanceOf[Locus].contig == i.end.asInstanceOf[Locus].contig)
            s"${i.start}-${i.end.asInstanceOf[Locus].position}"
          else
            s"${i.start}-${i.end}"
          s"${if (i.includesStart) "[" else "("}$bounds${if (i.includesEnd) "]" else ")"}"
        case _: TInterval =>
          JsonMethods.compact(t.toJSON(a))
        case TCall => Call.toString(a.asInstanceOf[Call])
        case _ => a.toString
      }
    }
  }
}

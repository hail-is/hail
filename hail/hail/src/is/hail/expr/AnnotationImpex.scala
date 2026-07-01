package is.hail.expr

import is.hail.annotations.{Annotation, NDArray, RowSeq, SafeNDArray}
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits._
import is.hail.types.physical.{
  PBoolean, PCanonicalArray, PCanonicalBinary, PCanonicalString, PCanonicalStruct, PFloat32,
  PFloat64, PInt32, PInt64, PType,
}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._

import scala.collection.compat._
import scala.collection.mutable

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s
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
      }.unsafeToArraySeq: _*)
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
  contigs: IndexedSeq[JSONExtractContig],
  xContigs: Set[String],
  yContigs: Set[String],
  mtContigs: Set[String],
  par: IndexedSeq[JSONExtractIntervalLocus],
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

object JSONAnnotationImpex extends Logging {
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
        case TFloat32 => JDouble(a.asInstanceOf[Float].toDouble)
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
          a match {
            case m: Map[_, _] =>
              JArray(m.map { case (k, v) =>
                JObject(
                  "key" -> exportAnnotation(k, keyType),
                  "value" -> exportAnnotation(v, valueType),
                )
              }.toList)
          }
        case TCall => JString(Call.toString(a.asInstanceOf[Call]))
        case TLocus(_) => a.asInstanceOf[Locus].toJSON
        case TInterval(pointType) => a.asInstanceOf[Interval].toJSON(pointType.export)
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
      RowSeq(importAnnotation(JsonMethods.parse(s), t, true, warnContext))
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

    def warnOnce(msg: String, parent: String): Unit =
      if (warnContext.add(parent)) logger.warn(msg)

    def warnCoerce(): Annotation = {
      warnOnce(s"Can't convert JSON value $jv to type $t at $parent.", parent)
      null
    }

    jv match {
      case JNull | JNothing => null
      case JInt(x) => t match {
          case TInt32 => x.toInt
          case TInt64 => x.toLong
          case TFloat32 => x.toFloat
          case TFloat64 => x.toDouble
          case TString => x.toString
          case _ => warnCoerce()
        }
      case JDouble(x) => t match {
          case TFloat64 => x
          case TFloat32 => x.toFloat
          case _ => warnCoerce()
        }
      case JString(x) => t match {
          case TFloat64 if doubleConv.contains(x) => doubleConv(x)
          case TFloat32 if floatConv.contains(x) => floatConv(x)
          case TString => x
          case TInt32 => x.toInt
          case TFloat64 =>
            if (x.startsWith("-:")) x.drop(2).toDouble
            else x.toDouble
          case TCall => Call.parse(x)
          case _ => warnCoerce()
        }
      case JBool(x) => t match {
          case TBoolean => x
          case _ => warnCoerce()
        }
      case JObject(jfields) => t match {
          // back compatibility
          case TDict(TString, valueType) =>
            jfields
              .foldLeft(Map.newBuilder[String, Annotation]) {
                (b, f) => b += (f._1 -> imp(f._2, valueType, parent))
              }
              .result()

          case t: TStruct =>
            if (t.size == 0) Annotation.empty
            else {
              val annotationSize =
                if (padNulls) t.size
                else if (jfields.isEmpty) 0
                else 1 + jfields.foldLeft(-1) { (max, f) =>
                  t.selfField(f._1).fold(max)(k => math.max(max, k.index))
                }

              val a = Array.fill[Any](annotationSize)(null)

              for ((name, jv2) <- jfields) {
                t.selfField(name) match {
                  case Some(f) =>
                    a(f.index) = imp(jv2, f.typ, parent + "." + name)
                  case None =>
                    warnOnce(
                      s"$t has no field $name at $parent for value $jv2",
                      parent + "/" + name,
                    )
                }
              }

              Annotation.fromSeq(ArraySeq.unsafeWrapArray(a))
            }
          case t @ TNDArray(_, _) =>
            jfields match {
              case List(("shape", shapeJson: JArray), ("data", dataJson: JArray)) =>
                val shapeArray =
                  shapeJson.arr.view.map(imp(_, TInt64, parent).asInstanceOf[Long]).to(ArraySeq)
                val dataArray = dataJson.arr.view.map(imp(_, t.elementType, parent)).to(ArraySeq)
                SafeNDArray(shapeArray, dataArray)
              case _ => warnCoerce()
            }
          case TInterval(pointType) =>
            val m = jfields.toMap
            val interval: Option[Annotation] =
              for {
                s <- m.get("start")
                start = imp(s, pointType, parent + ".start")

                e <- m.get("end")
                end = imp(e, pointType, parent + ".end")

                is <- m.get("includeStart")
                includesStart = imp(is, TBoolean, parent + ".includeStart").asInstanceOf[Boolean]

                ie <- m.get("includeEnd")
                includesEnd = imp(ie, TBoolean, parent + ".includeEnd").asInstanceOf[Boolean]
              } yield Interval(start, end, includesStart, includesEnd)

            interval.getOrElse(warnCoerce())

          case TLocus(_) => jv.extract[Locus]
          case _ => warnCoerce()
        }
      case JArray(elts) => t match {
          case TDict(keyType, valueType) =>
            val keyPath = parent + "[key]"
            val valuePath = parent + "[value]"
            elts.foldLeft(Map.newBuilder[Annotation, Annotation]) { (b, elem) =>
              elem match {
                case JObject(List(("key", k), ("value", v))) =>
                  b += (imp(k, keyType, keyPath) -> imp(v, valueType, valuePath))
                case _ =>
                  warnCoerce()
                  b
              }
            }.result()
          case t: TTuple =>
            if (t.size == 0) Annotation.empty
            else {
              val b = ArraySeq.newBuilder[Any]
              b.sizeHint(t.size)

              var i = 0
              for (e <- elts) {
                b += imp(e, t.types(i), parent)
                i += 1
              }

              if (padNulls && t.size > i) {
                b.sizeHint(t.size)
                for (_ <- i until t.size) b += null
              }

              Annotation.fromSeq(b.result())
            }
          case TArray(elementType) =>
            elts.view.map(jv2 => imp(jv2, elementType, parent + "[element]")).to(ArraySeq)
          case TSet(elementType) =>
            elts.view.map(jv2 => imp(jv2, elementType, parent + "[element]")).toSet
          case _ => warnCoerce()
        }
      case _ => warnCoerce()
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
        case t: TContainer => JsonMethods.compact(t.export(a))
        case t: TBaseStruct => JsonMethods.compact(t.export(a))
        case t: TNDArray => JsonMethods.compact(t.export(a))
        case TInterval(TLocus(_)) =>
          val i = a.asInstanceOf[Interval]
          val bounds = if (i.start.asInstanceOf[Locus].contig == i.end.asInstanceOf[Locus].contig)
            s"${i.start}-${i.end.asInstanceOf[Locus].position}"
          else
            s"${i.start}-${i.end}"
          s"${if (i.includesStart) "[" else "("}$bounds${if (i.includesEnd) "]" else ")"}"
        case _: TInterval =>
          JsonMethods.compact(t.export(a))
        case TCall => Call.toString(a.asInstanceOf[Call])
        case _ => a.toString
      }
    }
  }
}

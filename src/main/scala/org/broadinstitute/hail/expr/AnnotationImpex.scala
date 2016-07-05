package org.broadinstitute.hail.expr

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Sample, Variant}
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

abstract class AnnotationImpex[T, A] {
  // FIXME for now, schema must be specified on import
  def exportType(t: Type): T

  def exportAnnotation(a: Annotation, t: Type): A

  def importAnnotation(a: A, t: Type): Annotation
}

object SparkAnnotationImpex extends AnnotationImpex[DataType, Any] {
  val invalidCharacters = " ,;{}()\n\t=".toSet

  def escapeColumnName(name: String): String = {
    name.map { c =>
      if (invalidCharacters.contains(c))
        "_"
      else
        c
    }.mkString
  }

  def requiresConversion(t: Type): Boolean = t match {
    case TArray(elementType) => requiresConversion(elementType)
    case TSet(_) | TDict(_) | TGenotype | TAltAllele | TVariant => true
    case TStruct(fields) =>
      fields.exists(f => requiresConversion(f.`type`))
    case _ => false
  }

  def importType(t: DataType): Type = t match {
    case BooleanType => TBoolean
    case IntegerType => TInt
    case LongType => TLong
    case FloatType => TFloat
    case DoubleType => TDouble
    case StringType => TString
    case BinaryType => TBinary
    case ArrayType(elementType, _) => TArray(importType(elementType))
    case StructType(fields) =>
      TStruct(fields.zipWithIndex
        .map { case (f, i) =>
          (f.name, importType(f.dataType))
        }: _*)
  }

  def importAnnotation(a: Any, t: Type): Annotation = {
    if (a == null)
      null
    else
      t match {
        case TArray(elementType) =>
          a.asInstanceOf[Seq[_]].map(elem => importAnnotation(elem, elementType))
        case TSet(elementType) =>
          a.asInstanceOf[Seq[_]].map(elem => importAnnotation(elem, elementType)).toSet
        case TDict(elementType) =>
          val kvPairs = a.asInstanceOf[IndexedSeq[Annotation]]
          kvPairs
            .map(_.asInstanceOf[Row])
            .map(r => (r.get(0), importAnnotation(r.get(1), elementType)))
            .toMap
        case TGenotype =>
          val r = a.asInstanceOf[Row]
          Genotype(Option(r.get(0)).map(_.asInstanceOf[Int]),
            Option(r.get(1)).map(_.asInstanceOf[Seq[Int]].toArray),
            Option(r.get(2)).map(_.asInstanceOf[Int]),
            Option(r.get(3)).map(_.asInstanceOf[Int]),
            Option(r.get(4)).map(_.asInstanceOf[Seq[Int]].toArray),
            r.get(5).asInstanceOf[Boolean])
        case TAltAllele =>
          val r = a.asInstanceOf[Row]
          AltAllele(r.getAs[String](0), r.getAs[String](1))
        case TVariant =>
          val r = a.asInstanceOf[Row]
          Variant(r.getAs[String](0), r.getAs[Int](1), r.getAs[String](2),
            r.getAs[Seq[Row]](3).map(aa =>
              importAnnotation(aa, TAltAllele).asInstanceOf[AltAllele]).toArray)
        case TStruct(fields) =>
          val r = a.asInstanceOf[Row]
          Annotation.fromSeq(r.toSeq.zip(fields).map { case (v, f) =>
            importAnnotation(v, f.`type`)
          })
        case _ => a
      }
  }

  def exportType(t: Type): DataType = t match {
    case TBoolean => BooleanType
    case TInt => IntegerType
    case TLong => LongType
    case TFloat => FloatType
    case TDouble => DoubleType
    case TString => StringType
    case TChar => StringType
    case TBinary => BinaryType
    case TArray(elementType) => ArrayType(exportType(elementType))
    case TSet(elementType) => ArrayType(exportType(elementType))
    case TDict(elementType) =>
      ArrayType(StructType(Array(
        StructField("key", StringType, nullable = false),
        StructField("value", elementType.schema))))
    case TSample => StringType
    case TAltAllele => AltAllele.schema
    case TVariant => Variant.schema
    case TGenotype => Genotype.schema
    case TStruct(fields) =>
      if (fields.isEmpty)
        BooleanType //placeholder
      else
        StructType(fields
          .map(f =>
            StructField(escapeColumnName(f.name), f.`type`.schema)))
  }

  def exportAnnotation(a: Annotation, t: Type): Any = {
    if (a == null)
      null
    else
      t match {
        case TArray(elementType) =>
          a.asInstanceOf[IndexedSeq[_]].map(elem => exportAnnotation(elem, elementType))
        case TSet(elementType) =>
          a.asInstanceOf[Set[_]].toSeq.map(elem => exportAnnotation(elem, elementType))
        case TDict(elementType) =>
          a.asInstanceOf[Map[_, _]]
            .map { case (k, v) =>
              Row.fromSeq(Seq(k, exportAnnotation(v, elementType)))
            }.toIndexedSeq
        case TGenotype =>
          val g = a.asInstanceOf[Genotype]
          Row(g.gt.orNull, g.ad.map(_.toSeq).orNull, g.dp.orNull, g.gq.orNull, g.pl.map(_.toSeq).orNull, g.fakeRef)
        case TAltAllele =>
          val aa = a.asInstanceOf[AltAllele]
          Row(aa.ref, aa.alt)
        case TVariant =>
          val v = a.asInstanceOf[Variant]
          Row(v.contig, v.start, v.ref, v.altAlleles.map(aa => Row(aa.ref, aa.alt)))
        case TStruct(fields) =>
          val r = a.asInstanceOf[Row]
          Annotation.fromSeq(r.toSeq.zip(fields).map {
            case (v, f) => exportAnnotation(v, f.`type`)
          })
        case _ => a
      }
  }
}

case class JSONExtractGenotype(
  gt: Option[Int],
  ad: Option[Array[Int]],
  dp: Option[Int],
  gq: Option[Int],
  pl: Option[Array[Int]],
  fakeRef: Boolean) {
  def toGenotype =
    Genotype(gt, ad, dp, gq, pl, fakeRef)
}

case class JSONExtractVariant(contig: String,
  start: Int,
  ref: String,
  altAlleles: List[AltAllele]) {
  def toVariant =
    Variant(contig, start, ref, altAlleles.toArray)
}

object JSONAnnotationImpex extends AnnotationImpex[Type, JValue] {
  def jsonExtractVariant(t: Type, variantFields: String): Any => Option[Variant] = {
    val ec = EvalContext(Map(
      "root" -> (0, t)))

    val fs: Array[(BaseType, () => Option[Any])] = Parser.parseExprs(variantFields, ec)

    if (fs.length != 4)
      fatal(s"wrong number of variant field expressions: expected 4, got ${fs.length}")

    if (fs(0)._1 != TString)
      fatal(s"wrong type for chromosome field: expected String, got ${fs(0)._1}")
    if (fs(1)._1 != TInt)
      fatal(s"wrong type for pos field: expected Int, got ${fs(1)._1}")
    if (fs(2)._1 != TString)
      fatal(s"wrong type for ref field: expected String, got ${fs(2)._1}")
    if (fs(3)._1 != TArray(TString))
      fatal(s"wrong type for alt field: expected Array[String], got ${fs(3)._1}")

    (root: Annotation) => {
      ec.setAll(root)

      val vfs = fs.map(_._2())

      vfs(0).flatMap { chr =>
        vfs(1).flatMap { pos =>
          vfs(2).flatMap { ref =>
            vfs(3).map { alt =>
              Variant(chr.asInstanceOf[String],
                pos.asInstanceOf[Int],
                ref.asInstanceOf[String],
                alt.asInstanceOf[IndexedSeq[String]].toArray)
            }
          }
        }
      }
    }
  }

  def jsonExtractSample(t: Type, sampleExpr: String): Any => Option[String] = {
    val ec = EvalContext(Map(
      "root" -> (0, t)))

    val f: () => Option[Any] = Parser.parse(sampleExpr, ec, TString)

    (root: Annotation) => {
      ec.setAll(root)
      f().map(_.asInstanceOf[String])
    }
  }

  def exportType(t: Type): Type = t

  def exportAnnotation(a: Annotation, t: Type): JValue =
    if (a == null)
      JNull
    else {
      t match {
        case TBoolean => JBool(a.asInstanceOf[Boolean])
        case TChar => JString(a.asInstanceOf[String])
        case TInt => JInt(a.asInstanceOf[Int])
        case TLong => JInt(a.asInstanceOf[Long])
        case TFloat => JDouble(a.asInstanceOf[Float])
        case TDouble => JDouble(a.asInstanceOf[Double])
        case TString => JString(a.asInstanceOf[String])
        case TArray(elementType) =>
          val arr = a.asInstanceOf[Seq[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType)).toList)
        case TSet(elementType) =>
          val arr = a.asInstanceOf[Set[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType)).toList)
        case TDict(elementType) =>
          val m = a.asInstanceOf[Map[_, _]]
          JObject(m.map { case (k, v) =>
            (k.asInstanceOf[String], exportAnnotation(v, elementType))
          }.toList)
        case TSample => a.asInstanceOf[Sample].toJSON
        case TGenotype => a.asInstanceOf[Genotype].toJSON
        case TAltAllele => a.asInstanceOf[AltAllele].toJSON
        case TVariant => a.asInstanceOf[Variant].toJSON
        case TStruct(fields) =>
          val row = a.asInstanceOf[Row]
          JObject(fields
            .map(f => (f.name, exportAnnotation(row.get(f.index), f.`type`)))
            .toList)
      }
    }

  def importAnnotation(jv: JValue, t: Type): Annotation =
    importAnnotation(jv, t, "<root>")

  def importAnnotation(jv: JValue, t: Type, parent: String): Annotation = {
    implicit val formats = Serialization.formats(NoTypeHints)

    (jv, t) match {
      case (JNull | JNothing, _) => null
      case (JInt(x), TInt) => x.toInt
      case (JInt(x), TLong) => x.toLong
      case (JInt(x), TDouble) => x.toDouble
      case (JInt(x), TString) => x.toString
      case (JDouble(x), TDouble) => x
      case (JString("Infinity"), TDouble) => Double.PositiveInfinity
      case (JString("-Infinity"), TDouble) => Double.NegativeInfinity
      case (JString("Infinity"), TFloat) => Float.PositiveInfinity
      case (JString("-Infinity"), TFloat) => Float.NegativeInfinity
      case (JDouble(x), TFloat) => x.toFloat
      case (JString(x), TString) => x
      case (JString(x), TChar) => x
      case (JString(x), TInt) =>
        x.toInt
      case (JString(x), TDouble) =>
        if (x.startsWith("-:"))
          x.drop(2).toDouble
        else
          x.toDouble
      case (JBool(x), TBoolean) => x

      case (JObject(a), TDict(elementType)) =>
        a.map { case (key, value) =>
          (key, importAnnotation(value, elementType, parent))
        }
          .toMap

      case (JObject(jfields), t: TStruct) =>
        if (t.size == 0)
          Annotation.empty
        else {
          val a = Array.fill[Any](t.size)(null)

          for ((name, jv2) <- jfields) {
            t.selfField(name) match {
              case Some(f) =>
                a(f.index) = importAnnotation(jv2, f.`type`, parent + "." + name)

              case None =>
                warn(s"$t has no field $name at $parent")
            }
          }

          Annotation(a: _*)
        }
      case (_, TAltAllele) =>
        jv.extract[AltAllele]
      case (_, TVariant) =>
        jv.extract[JSONExtractVariant].toVariant
      case (_, TGenotype) =>
        jv.extract[JSONExtractGenotype].toGenotype

      case (JArray(a), TArray(elementType)) =>
        a.iterator.map(jv2 => importAnnotation(jv2, elementType, parent + ".<array>")).toArray[Any]: IndexedSeq[Any]

      case (JArray(a), TSet(elementType)) =>
        a.iterator.map(jv2 => importAnnotation(jv2, elementType, parent + ".<array>")).toSet[Any]

      case _ =>
        warn(s"Can't convert JSON value $jv to type $t at $parent.")
        null
    }
  }
}

object TableAnnotationImpex extends AnnotationImpex[Unit, String] {

  private val sb = new StringBuilder

  // Tables have no schema
  def exportType(t: Type): Unit = ()

  def exportAnnotation(a: Annotation, t: Type): String = {
    if (a == null)
      "NA"
    else {
      t match {
        case TDouble => a.asInstanceOf[Double].formatted("%.4e")
        case TString => a.asInstanceOf[String]
        case d: TDict => JsonMethods.compact(d.toJSON(a))
        case it: TIterable => JsonMethods.compact(it.toJSON(a))
        case t: TStruct => JsonMethods.compact(t.toJSON(a))
        case _ => a.toString
      }
    }
  }

  def importAnnotation(a: String, t: Type): Annotation = {
    (t: @unchecked) match {
      case TString => a
      case TInt => a.toInt
      case TLong => a.toLong
      case TFloat => a.toFloat
      case TDouble => a.toDouble
      case TBoolean => a.toBoolean
      case TVariant => a.split(":") match {
        case Array(chr, pos, ref, alt) => Variant(chr, pos.toInt, ref, alt.split(","))
      }
      case TAltAllele => a.split("/") match {
        case Array(ref, alt) => AltAllele(ref, alt)
      }
      case TGenotype => a.split(":").map(x => if (x == "." || x == "./.") None else Some(x)) match {
        case Array(gtStr, adStr, dpStr, gqStr, plStr) =>
          val gt = gtStr.map { gt =>
            val Array(gti, gtj) = gt.split("/").map(_.toInt)
            Genotype.gtIndex(gti, gtj)
          }
          val ad = adStr.map(_.split(",").map(_.toInt))
          val dp = dpStr.map(_.toInt)
          val gq = gqStr.map(_.toInt)
          val pl = plStr.map(_.split(",").map(_.toInt))
          Genotype(gt, ad, dp, gq, pl, false)
      }
      case TChar => a
      case t: TArray => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TSet => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TDict => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
      case t: TStruct => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t)
    }
  }
}

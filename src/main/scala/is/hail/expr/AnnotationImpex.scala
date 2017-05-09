package is.hail.expr

import is.hail.annotations.Annotation
import is.hail.utils.{Interval, _}
import is.hail.variant.{AltAllele, Contig, GenomeReference, Genotype, Locus, Sample, Variant}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

abstract class AnnotationImpex[T, A] {
  // FIXME for now, schema must be specified on import
  def exportType(t: Type): T

  def exportAnnotation(a: Annotation, t: Type, gr: GenomeReference): A

  def importAnnotation(a: A, t: Type, gr: GenomeReference): Annotation
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
    case TArray(elementType) => requiresConversion(elementType)
    case TSet(_) | TDict(_, _) | TGenotype | TAltAllele | TVariant | TLocus | TInterval => true
    case TStruct(fields) =>
      fields.isEmpty || fields.exists(f => requiresConversion(f.typ))
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

  def importAnnotation(a: Any, t: Type, gr: GenomeReference): Annotation = {
    if (a == null)
      null
    else
      t match {
        case TArray(elementType) =>
          a.asInstanceOf[Seq[_]].map(elem => importAnnotation(elem, elementType, gr))
        case TSet(elementType) =>
          a.asInstanceOf[Seq[_]].map(elem => importAnnotation(elem, elementType, gr)).toSet
        case TDict(keyType, valueType) =>
          val kvPairs = a.asInstanceOf[IndexedSeq[Annotation]]
          kvPairs
            .map(_.asInstanceOf[Row])
            .map(r => (importAnnotation(r.get(0), keyType, gr), importAnnotation(r.get(1), valueType, gr)))
            .toMap
        case TGenotype =>
          val r = a.asInstanceOf[Row]
          Genotype(Option(r.get(0)).map(_.asInstanceOf[Int]),
            Option(r.get(1)).map(_.asInstanceOf[Seq[Int]].toArray),
            Option(r.get(2)).map(_.asInstanceOf[Int]),
            Option(r.get(3)).map(_.asInstanceOf[Int]),
            Option(r.get(4)).map(_.asInstanceOf[Seq[Int]].toArray),
            r.get(5).asInstanceOf[Boolean],
            r.get(6).asInstanceOf[Boolean])
        case TAltAllele =>
          val r = a.asInstanceOf[Row]
          AltAllele(r.getAs[String](0), r.getAs[String](1))
        case TVariant =>
          val r = a.asInstanceOf[Row]
          Variant(r.getAs[String](0), r.getAs[Int](1), r.getAs[String](2),
            r.getAs[Seq[Row]](3).map(aa =>
              importAnnotation(aa, TAltAllele, gr).asInstanceOf[AltAllele]).toArray)(gr)
        case TLocus =>
          val r = a.asInstanceOf[Row]
          Locus(r.getAs[String](0), r.getAs[Int](1))(gr)
        case TInterval =>
          val r = a.asInstanceOf[Row]
          Interval(importAnnotation(r.get(0), TLocus, gr).asInstanceOf[Locus], importAnnotation(r.get(1), TLocus, gr).asInstanceOf[Locus])
        case TStruct(fields) =>
          if (fields.isEmpty)
            null
          else {
            val r = a.asInstanceOf[Row]
            Annotation.fromSeq(r.toSeq.zip(fields).map { case (v, f) =>
              importAnnotation(v, f.typ, gr)
            })
          }
        case _ => a
      }
  }

  def exportType(t: Type): DataType = (t: @unchecked) match {
    case TBoolean => BooleanType
    case TInt => IntegerType
    case TLong => LongType
    case TFloat => FloatType
    case TDouble => DoubleType
    case TString => StringType
    case TBinary => BinaryType
    case TArray(elementType) => ArrayType(exportType(elementType))
    case TSet(elementType) => ArrayType(exportType(elementType))
    case TDict(keyType, valueType) =>
      ArrayType(StructType(Array(
        StructField("key", keyType.schema),
        StructField("value", valueType.schema))))
    case TAltAllele => AltAllele.sparkSchema
    case TVariant => Variant.sparkSchema
    case TLocus => Locus.sparkSchema
    case TInterval => StructType(Array(
      StructField("start", Locus.sparkSchema, nullable = false),
      StructField("end", Locus.sparkSchema, nullable = false)))
    case TGenotype => Genotype.sparkSchema
    case TCall => IntegerType
    case TStruct(fields) =>
      if (fields.isEmpty)
        BooleanType //placeholder
      else
        StructType(fields
          .map(f =>
            StructField(escapeColumnName(f.name), f.typ.schema)))
  }

  def exportAnnotation(a: Annotation, t: Type, gr: GenomeReference): Any = {
    if (a == null)
      null
    else
      t match {
        case TArray(elementType) =>
          a.asInstanceOf[IndexedSeq[_]].map(elem => exportAnnotation(elem, elementType, gr))
        case TSet(elementType) =>
          a.asInstanceOf[Set[_]].toSeq.map(elem => exportAnnotation(elem, elementType, gr))
        case TDict(keyType, valueType) =>
          a.asInstanceOf[Map[_, _]]
            .map { case (k, v) =>
              Row.fromSeq(Seq(exportAnnotation(k, keyType, gr), exportAnnotation(v, valueType, gr)))
            }.toIndexedSeq
        case TGenotype =>
          val g = a.asInstanceOf[Genotype]
          g.toRow
        case TAltAllele =>
          val aa = a.asInstanceOf[AltAllele]
          Row(aa.ref, aa.alt)
        case TVariant =>
          val v = a.asInstanceOf[Variant]
          Row(v.contigStr(gr), v.start, v.ref, v.altAlleles.map(aa => Row(aa.ref, aa.alt)))
        case TLocus =>
          val l = a.asInstanceOf[Locus]
          Row(l.contigStr(gr), l.position)
        case TInterval =>
          val i = a.asInstanceOf[Interval[_]]
          Row(exportAnnotation(i.start, TLocus, gr), exportAnnotation(i.end, TLocus, gr))
        case TStruct(fields) =>
          if (fields.isEmpty)
            null
          else {
            val r = a.asInstanceOf[Row]
            Annotation.fromSeq(r.toSeq.zip(fields).map {
              case (v, f) => exportAnnotation(v, f.typ, gr)
            })
          }
        case _ => a
      }
  }
}

case class JSONExtractGenotype(
  gt: Option[Int],
  ad: Option[Array[Int]],
  dp: Option[Int],
  gq: Option[Int],
  px: Option[Array[Int]],
  fakeRef: Boolean,
  isLinearScale: Boolean) {
  def toGenotype =
    Genotype(gt, ad, dp, gq, px, fakeRef, isLinearScale)
}

case class JSONExtractVariant(contig: String,
  start: Int,
  ref: String,
  altAlleles: List[AltAllele]) {
  def toVariant(gr: GenomeReference) =
    Variant(contig, start, ref, altAlleles.toArray)(gr)
}

case class JSONExtractLocus(contig: String,
  position: Int) {
  def toLocus(gr: GenomeReference) =
    Locus(contig, position)(gr)
}

case class JSONExtractInterval(start: JSONExtractLocus, end: JSONExtractLocus) {
  def toInterval(gr: GenomeReference) = Interval(start.toLocus(gr), end.toLocus(gr))
}

case class JSONExtractGenomeReference(name: String, contigs: Array[Contig], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[JSONExtractInterval]) {

  def toGenomeReference: GenomeReference = {
    val gr = GenomeReference(name, contigs, xContigs, yContigs, mtContigs)
    GenomeReference.setReference(gr)
    gr.par = par.map(_.toInterval(gr))
    gr
  }
}

object JSONAnnotationImpex extends AnnotationImpex[Type, JValue] {
  def jsonExtractVariant(t: Type, variantFields: String, gr: GenomeReference): Any => Variant = {
    val ec = EvalContext(Map(
      "root" -> (0, t)))

    val (types, f) = Parser.parseExprs(variantFields, ec)

    if (types.length != 4)
      fatal(s"wrong number of variant field expressions: expected 4, got ${ types.length }")

    if (types(0) != TString)
      fatal(s"wrong type for chromosome field: expected String, got ${ types(0) }")
    if (types(1) != TInt)
      fatal(s"wrong type for pos field: expected Int, got ${ types(1) }")
    if (types(2) != TString)
      fatal(s"wrong type for ref field: expected String, got ${ types(2) }")
    if (types(3) != TArray(TString))
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
          alt.asInstanceOf[IndexedSeq[String]].toArray)(gr)
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

  def exportAnnotation(a: Annotation, t: Type, gr: GenomeReference): JValue =
    if (a == null)
      JNull
    else {
      (t: @unchecked) match {
        case TBoolean => JBool(a.asInstanceOf[Boolean])
        case TInt => JInt(a.asInstanceOf[Int])
        case TLong => JInt(a.asInstanceOf[Long])
        case TFloat => JDouble(a.asInstanceOf[Float])
        case TDouble => JDouble(a.asInstanceOf[Double])
        case TString => JString(a.asInstanceOf[String])
        case TArray(elementType) =>
          val arr = a.asInstanceOf[Seq[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType, gr)).toList)
        case TSet(elementType) =>
          val arr = a.asInstanceOf[Set[Any]]
          JArray(arr.map(elem => exportAnnotation(elem, elementType, gr)).toList)
        case TDict(keyType, valueType) =>
          val m = a.asInstanceOf[Map[_, _]]
          JArray(m.map { case (k, v) => JObject(
            "key" -> exportAnnotation(k, keyType, gr),
            "value" -> exportAnnotation(v, valueType, gr))
          }.toList)
        case TCall => JInt(a.asInstanceOf[Int])
        case TGenotype => a.asInstanceOf[Genotype].toJSON
        case TAltAllele => a.asInstanceOf[AltAllele].toJSON
        case TVariant => a.asInstanceOf[Variant].toJSON(gr)
        case TLocus => a.asInstanceOf[Locus].toJSON(gr)
        case TInterval => a.asInstanceOf[Interval[Locus]].toJSON(TLocus.toJSON(_, gr))
        case TStruct(fields) =>
          val row = a.asInstanceOf[Row]
          JObject(fields
            .map(f => (f.name, exportAnnotation(row.get(f.index), f.typ, gr)))
            .toList)
      }
    }

  def importAnnotation(jv: JValue, t: Type, gr: GenomeReference): Annotation =
    importAnnotation(jv, t, "<root>", gr)

  def importAnnotation(jv: JValue, t: Type, parent: String, gr: GenomeReference): Annotation = {
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
      case (JString(x), TInt) =>
        x.toInt
      case (JString(x), TDouble) =>
        if (x.startsWith("-:"))
          x.drop(2).toDouble
        else
          x.toDouble
      case (JBool(x), TBoolean) => x

      // back compatibility
      case (JObject(a), TDict(TString, valueType)) =>
        a.map { case (key, value) =>
          (key, importAnnotation(value, valueType, parent, gr))
        }
          .toMap

      case (JArray(arr), TDict(keyType, valueType)) =>
        arr.map { case JObject(a) =>
          a match {
            case List(k, v) =>
              (k, v) match {
                case (("key", ka), ("value", va)) =>
                  (importAnnotation(ka, keyType, parent, gr), importAnnotation(va, valueType, parent, gr))
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
                a(f.index) = importAnnotation(jv2, f.typ, parent + "." + name, gr)

              case None =>
                warn(s"$t has no field $name at $parent")
            }
          }

          Annotation(a: _*)
        }
      case (_, TAltAllele) =>
        jv.extract[AltAllele]
      case (_, TVariant) =>
        jv.extract[JSONExtractVariant].toVariant(gr)
      case (_, TLocus) =>
        jv.extract[JSONExtractLocus].toLocus(gr)
      case (_, TInterval) =>
        jv.extract[JSONExtractInterval].toInterval(gr)
      case (_, TGenotype) =>
        jv.extract[JSONExtractGenotype].toGenotype
      case (JInt(x), TCall) => x.toInt

      case (JArray(a), TArray(elementType)) =>
        a.iterator.map(jv2 => importAnnotation(jv2, elementType, parent + ".<array>", gr)).toArray[Any]: IndexedSeq[Any]

      case (JArray(a), TSet(elementType)) =>
        a.iterator.map(jv2 => importAnnotation(jv2, elementType, parent + ".<array>", gr)).toSet[Any]

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

  def exportAnnotation(a: Annotation, t: Type, gr: GenomeReference): String = {
    if (a == null)
      "NA"
    else {
      t match {
        case TDouble => a.asInstanceOf[Double].formatted("%.4e")
        case TString => a.asInstanceOf[String]
        case d: TDict => JsonMethods.compact(d.toJSON(a, gr))
        case it: TIterable => JsonMethods.compact(it.toJSON(a, gr))
        case t: TStruct => JsonMethods.compact(t.toJSON(a, gr))
        case TGenotype => JsonMethods.compact(t.toJSON(a, gr))
        case TInterval =>
          val i = a.asInstanceOf[Interval[Locus]]
          if (i.start.contig == i.end.contig)
            s"${ i.start.toString(gr) }-${ i.end.position }"
          else s"${ i.start.toString(gr) }-${ i.end.toString(gr) }"
        case TLocus => a.asInstanceOf[Locus].toString(gr)
        case TVariant => a.asInstanceOf[Variant].toString(gr)
        case _ => a.toString
      }
    }
  }

  def importAnnotation(a: String, t: Type, gr: GenomeReference): Annotation = {
    (t: @unchecked) match {
      case TString => a
      case TInt => a.toInt
      case TLong => a.toLong
      case TFloat => a.toFloat
      case TDouble => if (a == "nan") Double.NaN else a.toDouble
      case TBoolean => a.toBoolean
      case TLocus => Locus.parse(a, gr)
      case TInterval => Locus.parseInterval(a)(gr)
      case TVariant => Variant.parse(a)(gr)
      case TAltAllele => a.split("/") match {
        case Array(ref, alt) => AltAllele(ref, alt)
      }
      case TGenotype => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t, gr)
      case TCall => a.toInt
      case t: TArray => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t, gr)
      case t: TSet => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t, gr)
      case t: TDict => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t, gr)
      case t: TStruct => JSONAnnotationImpex.importAnnotation(JsonMethods.parse(a), t, gr)
    }
  }
}

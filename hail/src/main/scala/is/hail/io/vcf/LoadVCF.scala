package is.hail.io.vcf

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{BroadcastValue, ExecuteContext, HailStateManager}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.{
  CloseableIterator, EmitCode, EmitCodeBuilder, EmitMethodBuilder, GenericLine, GenericLines,
  GenericTableValue, IEmitCode, IR, IRParser, Literal, LowerMatrixIR, MatrixHybridReader,
  MatrixReader, PartitionReader,
}
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.{VCFAttributes, VCFMetadata}
import is.hail.io.fs.{FS, FileListEntry}
import is.hail.io.tabix._
import is.hail.io.vcf.LoadVCF.{getHeaderLines, parseHeader}
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SStreamValue}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._

import scala.annotation.meta.param
import scala.annotation.switch
import scala.collection.JavaConverters._

import htsjdk.variant.vcf._
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JValue}
import org.json4s.JsonAST.{JArray, JObject, JString}
import org.json4s.jackson.JsonMethods

class BufferedLineIterator(bit: BufferedIterator[String])
    extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove(): Unit = {
    throw new UnsupportedOperationException
  }
}

object VCFHeaderInfo {
  val headerType: TStruct = TStruct(
    "sampleIDs" -> TArray(TString),
    "infoFields" -> TArray(TTuple(TString, TString)),
    "formatFields" -> TArray(TTuple(TString, TString)),
    "filterAttrs" -> TDict(TString, TDict(TString, TString)),
    "infoAttrs" -> TDict(TString, TDict(TString, TString)),
    "formatAttrs" -> TDict(TString, TDict(TString, TString)),
    "infoFlagFields" -> TArray(TString),
  )

  val headerTypePType: PType = PType.canonical(headerType, required = false, innerRequired = true)

  def fromJSON(jv: JValue): VCFHeaderInfo = {
    val sampleIDs =
      (jv \ "sampleIDs").asInstanceOf[JArray].arr.map(_.asInstanceOf[JString].s).toArray
    val infoFlagFields =
      (jv \ "infoFlagFields").asInstanceOf[JArray].arr.map(_.asInstanceOf[JString].s).toSet

    def lookupFields(name: String) = (jv \ name).asInstanceOf[JArray].arr.map { case elt: JArray =>
      val List(name: JString, typeStr: JString) = elt.arr
      name.s -> IRParser.parseType(typeStr.s)
    }.toArray

    val infoFields = lookupFields("infoFields")
    val formatFields = lookupFields("formatFields")

    def lookupAttrs(name: String) = (jv \ name).asInstanceOf[JObject].obj.toMap
      .mapValues { case elt: JObject =>
        elt.obj.toMap.mapValues(_.asInstanceOf[JString].s)
      }

    val filterAttrs = lookupAttrs("filterAttrs")
    val infoAttrs = lookupAttrs("infoAttrs")
    val formatAttrs = lookupAttrs("formatAttrs")
    VCFHeaderInfo(sampleIDs, infoFields, formatFields, filterAttrs, infoAttrs, formatAttrs,
      infoFlagFields)
  }
}

case class VCFHeaderInfo(
  sampleIds: Array[String],
  infoFields: Array[(String, Type)],
  formatFields: Array[(String, Type)],
  filtersAttrs: VCFAttributes,
  infoAttrs: VCFAttributes,
  formatAttrs: VCFAttributes,
  infoFlagFields: Set[String],
) {

  def formatCompatible(other: VCFHeaderInfo): Boolean = {
    val m = formatFields.toMap
    other.formatFields.forall { case (name, t) => m(name) == t }
  }

  def infoCompatible(other: VCFHeaderInfo): Boolean = {
    val m = infoFields.toMap
    other.infoFields.forall { case (name, t) => m.get(name).contains(t) }
  }

  def genotypeSignature: TStruct = TStruct(formatFields: _*)
  def infoSignature: TStruct = TStruct(infoFields: _*)

  def getPTypes(arrayElementsRequired: Boolean, entryFloatType: Type, callFields: Set[String])
    : (PStruct, PStruct, PStruct) = {

    def typeToPType(
      fdName: String,
      t: Type,
      floatType: Type,
      required: Boolean,
      isCallField: Boolean,
    ): PType = {
      t match {
        case TString if isCallField => PCanonicalCall(required)
        case t if isCallField =>
          fatal(s"field '$fdName': cannot parse type $t as call field")
        case TFloat64 =>
          PType.canonical(floatType, required)
        case TString =>
          PCanonicalString(required)
        case TInt32 =>
          PInt32(required)
        case TBoolean =>
          PBooleanRequired
        case TArray(t2) =>
          PCanonicalArray(
            typeToPType(fdName, t2, floatType, arrayElementsRequired, false),
            required,
          )
      }
    }

    val infoType = PCanonicalStruct(
      true,
      infoFields.map { case (name, t) =>
        (name, typeToPType(name, t, TFloat64, false, false))
      }: _*
    )
    val formatType = PCanonicalStruct(
      true,
      formatFields.map { case (name, t) =>
        (
          name,
          typeToPType(name, t, entryFloatType, false, name == "GT" || callFields.contains(name)),
        )
      }: _*
    )

    val vaSignature = PCanonicalStruct(
      Array(
        PField("rsid", PCanonicalString(), 0),
        PField("qual", PFloat64(), 1),
        PField("filters", PCanonicalSet(PCanonicalString(true)), 2),
        PField("info", infoType, 3),
      ),
      true,
    )
    (infoType, vaSignature, formatType)
  }

  def writeToRegion(sm: HailStateManager, r: Region, dropAttrs: Boolean): Long = {
    val rvb = new RegionValueBuilder(sm, r)
    rvb.start(VCFHeaderInfo.headerTypePType)
    rvb.startStruct()
    rvb.addAnnotation(rvb.currentType().virtualType, sampleIds.toFastSeq)
    rvb.addAnnotation(
      rvb.currentType().virtualType,
      infoFields.map { case (x1, x2) => Row(x1, x2.parsableString()) }.toFastSeq,
    )
    rvb.addAnnotation(
      rvb.currentType().virtualType,
      formatFields.map { case (x1, x2) => Row(x1, x2.parsableString()) }.toFastSeq,
    )
    rvb.addAnnotation(rvb.currentType().virtualType, if (dropAttrs) Map.empty else filtersAttrs)
    rvb.addAnnotation(rvb.currentType().virtualType, if (dropAttrs) Map.empty else infoAttrs)
    rvb.addAnnotation(rvb.currentType().virtualType, if (dropAttrs) Map.empty else formatAttrs)
    rvb.addAnnotation(rvb.currentType().virtualType, infoFlagFields.toFastSeq.sorted)
    rvb.result().offset
  }

  def toJSON: JValue = {
    def fieldsJson(fields: Array[(String, Type)]): JValue = JArray(fields.map { case (name, t) =>
      JArray(List(JString(name), JString(t.parsableString())))
    }.toList)

    def attrsJson(attrs: Map[String, Map[String, String]]): JValue = JObject(attrs.map {
      case (name, m) =>
        (name, JObject(name -> JObject(m.map { case (k, v) => (k, JString(v)) }.toList)))
    }.toList)

    JObject(
      "sampleIDs" -> JArray(sampleIds.map(JString).toList),
      "infoFields" -> fieldsJson(infoFields),
      "formatFields" -> fieldsJson(formatFields),
      "filtersAttrs" -> attrsJson(filtersAttrs),
      "infoAttrs" -> attrsJson(infoAttrs),
      "formatAttrs" -> attrsJson(formatAttrs),
      "infoFlagFields" -> JArray(infoFlagFields.map(JString).toList),
    )
  }
}

class VCFParseError(val msg: String, val pos: Int) extends RuntimeException(msg)

final class VCFLine(
  val line: String,
  val fileNum: Long,
  val fileOffset: Long,
  arrayElementsRequired: Boolean,
  val abs: MissingArrayBuilder[String],
  val abi: MissingArrayBuilder[Int],
  val abf: MissingArrayBuilder[Float],
  val abd: MissingArrayBuilder[Double],
) {
  var pos: Int = 0

  def parseError(msg: String): Unit = throw new VCFParseError(msg, pos)

  def numericValue(c: Char): Int = {
    if (c < '0' || c > '9')
      parseError(
        s"invalid character '${StringEscapeUtils.escapeString(c.toString)}' in integer literal"
      )
    c - '0'
  }

  /* field contexts: field, array field, format field, call field, format array field, filter array
   * field */

  def endField(p: Int): Boolean =
    p == line.length || line(p) == '\t'

  def endArrayElement(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ','
    }
  }

  def endInfoKey(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == '=' || c == ';'
    }
  }

  def endInfoField(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ';'
    }
  }

  def endFormatField(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ':'
    }
  }

  // field within call
  def endCallField(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ':' || c == '/' || c == '|'
    }
  }

  def endInfoArrayElement(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ';' || c == ','
    }
  }

  def endFormatArrayElement(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ':' || c == ','
    }
  }

  def endFilterArrayElement(p: Int): Boolean = endInfoField

  def endField(): Boolean = endField(pos)

  def endArrayElement(): Boolean = endArrayElement(pos)

  def endInfoField(): Boolean = endInfoField(pos)

  def endInfoKey(): Boolean = endInfoKey(pos)

  def endFormatField(): Boolean = endFormatField(pos)

  def endCallField(): Boolean = endCallField(pos)

  def endInfoArrayElement(): Boolean = endInfoArrayElement(pos)

  def endFormatArrayElement(): Boolean = endFormatArrayElement(pos)

  def endFilterArrayElement(): Boolean = endFilterArrayElement(pos)

  def skipInfoField(): Unit =
    while (!endInfoField())
      pos += 1

  def skipFormatField(): Unit =
    while (!endFormatField())
      pos += 1

  def fieldMissing(): Boolean =
    pos < line.length &&
      line(pos) == '.' &&
      endField(pos + 1)

  def arrayFieldMissing(): Boolean =
    pos < line.length &&
      line(pos) == '.' &&
      endArrayElement(pos + 1)

  def infoFieldMissing(): Boolean =
    pos < line.length &&
      (line(pos) == '.' &&
        endInfoField(pos + 1) ||
        endInfoField(pos))

  def formatFieldMissing(): Boolean =
    pos < line.length &&
      line(pos) == '.' &&
      endFormatField(pos + 1)

  def callFieldMissing(): Boolean =
    pos < line.length &&
      line(pos) == '.' &&
      endCallField(pos + 1)

  def infoArrayElementMissing(): Boolean =
    pos < line.length &&
      line(pos) == '.' &&
      endInfoArrayElement(pos + 1)

  def formatArrayElementMissing(): Boolean =
    pos < line.length &&
      line(pos) == '.' &&
      endFormatArrayElement(pos + 1)

  def parseString(): String = {
    val start = pos
    while (!endField())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseInt(): Int = {
    if (endField())
      parseError("empty integer literal")
    var mul = 1
    if (line(pos) == '-') {
      mul = -1
      pos += 1
    }
    var v = numericValue(line(pos))
    pos += 1
    while (!endField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v * mul
  }

  def skipField(): Unit =
    while (!endField())
      pos += 1

  def parseStringInArray(): String = {
    val start = pos
    while (!endArrayElement())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  // leaves result in abs
  def parseAltAlleles(): Unit = {
    assert(abs.size == 0)

    // . means no alternate alleles
    if (fieldMissing()) {
      pos += 1 // .
      return
    }

    abs += parseStringInArray()
    while (!endField()) {
      pos += 1 // comma
      abs += parseStringInArray()
    }
  }

  // leaves result in abs, returns true for having filters, even PASS, false for no filters
  def parseFilters(): Boolean = {
    def parseStringInFilters(): String = {
      val start = pos
      while (!endFilterArrayElement())
        pos += 1
      val end = pos
      line.substring(start, end)
    }

    assert(abs.size == 0)

    // . means no filters
    if (fieldMissing()) {
      pos += 1 // .
      false
    } else {
      val s = parseStringInFilters()
      if (!(s == "PASS" && endField())) {
        abs += s
        while (!endField()) {
          pos += 1 // semicolon
          abs += parseStringInFilters()
        }
      }
      true
    }
  }

  def nextField(): Unit = {
    if (pos == line.length)
      parseError("unexpected end of line")
    if (line(pos) != '\t') {
      parseError("expected tab character between fields")
    }
    pos += 1 // tab
  }

  def nextInfoField(): Unit = {
    if (pos == line.length)
      parseError("unexpected end of line")
    assert(line(pos) == ';')
    pos += 1 // semicolon
  }

  def nextFormatField(): Unit = {
    if (pos == line.length)
      parseError("unexpected end of line")
    assert(line(pos) == ':')
    pos += 1 // colon
  }

  def acceptableRefAlleleBases(ref: String): Boolean = {
    var i = 0
    var isStandardAllele: Boolean = true
    while (i < ref.length) {
      (ref(i): @switch) match {
        case 'A' | 'T' | 'G' | 'C' | 'a' | 't' | 'g' | 'c' | 'N' | 'n' =>
        case _ =>
          isStandardAllele = false
      }
      i += 1
    }
    return isStandardAllele || htsjdk.variant.variantcontext.Allele.acceptableAlleleBases(ref)
  }

  // return false if it should be filtered
  def parseAddVariant(
    rvb: RegionValueBuilder,
    rg: Option[ReferenceGenome],
    contigRecoding: Map[String, String],
    hasLocus: Boolean,
    hasAlleles: Boolean,
    hasRSID: Boolean,
    skipInvalidLoci: Boolean,
  ): Boolean = {
    assert(pos == 0)

    if (line.isEmpty || line(0) == '#')
      return false

    // CHROM (contig)
    val contig = parseString()
    val recodedContig = contigRecoding.getOrElse(contig, contig)
    nextField()

    // POS (start)
    val start = parseInt()
    nextField()

    if (skipInvalidLoci) {
      if (!rg.forall(_.isValidLocus(recodedContig, start)))
        return false
    } else
      rg.foreach(_.checkLocus(recodedContig, start))

    // ID
    val rsid = parseString()
    nextField()

    // REF
    val ref = parseString()
    if (!acceptableRefAlleleBases(ref))
      return false
    nextField()

    // ALT
    parseAltAlleles()
    nextField()

    if (hasLocus) {
      rg match {
        case Some(_) => rvb.addLocus(recodedContig, start)
        case None => // Without a reference genome, we use a struct of two fields rather than a PLocus
          rvb.startStruct() // pk: Locus
          rvb.addString(recodedContig)
          rvb.addInt(start)
          rvb.endStruct()
      }

    }

    if (hasAlleles) {
      rvb.startArray(abs.length + 1) // ref plus alts
      rvb.addString(ref)
      var i = 0
      while (i < abs.length) {
        rvb.addString(abs(i))
        i += 1
      }
      rvb.endArray()
    }

    if (hasRSID) {
      if (rsid == ".")
        rvb.setMissing()
      else
        rvb.addString(rsid)
    }

    abs.clear()

    true
  }

  def parseIntInCall(): Int = {
    if (endCallField())
      parseError("empty integer field")
    var v = numericValue(line(pos))
    pos += 1
    while (!endCallField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v
  }

  def parseAddCall(rvb: RegionValueBuilder): Unit = {
    if (pos == line.length)
      parseError("empty call")

    var j = 0
    var mj = false
    if (callFieldMissing()) {
      mj = true
      pos += 1
    } else
      j = parseIntInCall()

    if (endFormatField()) {
      // haploid
      if (mj)
        rvb.setMissing()
      else
        rvb.addCall(Call1(j, phased = false))
      return
    }

    if (line(pos) != '|' && line(pos) != '/')
      parseError("parse error in call")
    val isPhased = line(pos) == '|'
    pos += 1

    var k = 0
    var mk = false
    if (callFieldMissing()) {
      mk = true
      pos += 1
    } else
      k = parseIntInCall()

    if (!endFormatField()) {
      if (line(pos) == '/' || line(pos) == '|')
        parseError("ploidy > 2 not supported") // FIXME: Allow N-ploidy when supported
      else
        parseError("parse error in call")
    }

    // treat partially missing like missing
    if (mj || mk)
      rvb.setMissing()
    else {
      rvb.addCall(Call2(j, k, phased = isPhased))
    }
  }

  def parseFormatInt(): Int = {
    if (endFormatField())
      parseError("empty integer")
    var mul = 1
    if (line(pos) == '-') {
      mul = -1
      pos += 1
    }
    var v = numericValue(line(pos))
    pos += 1
    while (!endFormatField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v * mul
  }

  def parseAddFormatInt(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else
      rvb.addInt(parseFormatInt())
  }

  def parseFormatString(): String = {
    val start = pos
    while (!endFormatField())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseAddFormatString(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else
      rvb.addString(parseFormatString())
  }

  def parseFormatFloat(): Float = {
    val s = parseFormatString()
    VCFUtils.parseVcfDouble(s).toFloat
  }

  def parseAddFormatFloat(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      rvb.addFloat(parseFormatFloat())
    }
  }

  def parseFormatDouble(): Double = {
    val s = parseFormatString()
    VCFUtils.parseVcfDouble(s)
  }

  def parseAddFormatDouble(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else
      rvb.addDouble(parseFormatDouble())
  }

  def parseIntInFormatArray(): Int = {
    if (endFormatArrayElement())
      parseError("empty integer")
    var mul = 1
    if (line(pos) == '-') {
      mul = -1
      pos += 1
    }
    var v = numericValue(line(pos))
    pos += 1
    while (!endFormatArrayElement()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v * mul
  }

  def parseStringInFormatArray(): String = {
    val start = pos
    while (!endFormatArrayElement())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseFloatInFormatArray(): Float = {
    val s = parseStringInFormatArray()
    s.toFloat
  }

  def parseDoubleInFormatArray(): Double = {
    val s = parseStringInFormatArray()
    s.toDouble
  }

  def parseArrayElement[T](ab: MissingArrayBuilder[T], eltParser: () => T): Unit = {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in FORMAT array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      ab.addMissing()
      pos += 1
    } else {
      ab += eltParser()
    }
  }

  def parseArrayIntElement(): Unit = {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in FORMAT array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abi.addMissing()
      pos += 1
    } else {
      abi += parseIntInFormatArray()
    }
  }

  def parseFloatArrayElement(): Unit = {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in FORMAT array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abf.addMissing()
      pos += 1
    } else {
      abf += parseFloatInFormatArray()
    }
  }

  def parseArrayDoubleElement(): Unit = {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in FORMAT array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abd.addMissing()
      pos += 1
    } else {
      abd += parseDoubleInFormatArray()
    }
  }

  def parseArrayStringElement(): Unit = {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in FORMAT array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abs.addMissing()
      pos += 1
    } else {
      abs += parseStringInFormatArray()
    }
  }

  def parseAddFormatArrayInt(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abi.length == 0)

      parseArrayIntElement()

      while (!endFormatField()) {
        pos += 1 // comma
        parseArrayIntElement()
      }

      rvb.startArray(abi.length)
      var i = 0
      while (i < abi.length) {
        if (abi.isMissing(i))
          rvb.setMissing()
        else
          rvb.addInt(abi(i))
        i += 1
      }
      rvb.endArray()

      abi.clear()
    }
  }

  def parseAddFormatArrayString(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abs.length == 0)

      parseArrayStringElement()
      while (!endFormatField()) {
        pos += 1 // comma
        parseArrayStringElement()
      }

      rvb.startArray(abs.length)
      var i = 0
      while (i < abs.length) {
        rvb.addString(abs(i))
        i += 1
      }
      rvb.endArray()

      abs.clear()
    }
  }

  def parseAddFormatArrayFloat(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abf.length == 0)

      parseFloatArrayElement()
      while (!endFormatField()) {
        pos += 1 // comma
        parseFloatArrayElement()
      }

      rvb.startArray(abf.length)
      var i = 0
      while (i < abf.length) {
        if (abf.isMissing(i))
          rvb.setMissing()
        else
          rvb.addFloat(abf(i))
        i += 1
      }
      rvb.endArray()

      abf.clear()
    }
  }

  def parseAddFormatArrayDouble(rvb: RegionValueBuilder): Unit = {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abd.length == 0)

      parseArrayDoubleElement()
      while (!endFormatField()) {
        pos += 1 // comma
        parseArrayDoubleElement()
      }

      rvb.startArray(abd.length)
      var i = 0
      while (i < abd.length) {
        if (abd.isMissing(i))
          rvb.setMissing()
        else
          rvb.addDouble(abd(i))
        i += 1
      }
      rvb.endArray()

      abd.clear()
    }
  }

  def parseInfoKey(): String = {
    val start = pos
    while (!endInfoKey()) {
      if (line(pos) == ' ')
        parseError("space character in INFO key")
      pos += 1
    }
    val end = pos
    line.substring(start, end)
  }

  def parseInfoInt(): Int = {
    if (endInfoField())
      parseError("empty integer")
    var mul = 1
    if (line(pos) == '-') {
      mul = -1
      pos += 1
    }
    var v = 0
    while (!endInfoField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v * mul
  }

  def parseAddInfoInt(rvb: RegionValueBuilder): Unit = {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      rvb.addInt(parseInfoInt())
    }
  }

  def parseInfoString(): String = {
    val start = pos
    while (!endInfoField())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseAddInfoString(rvb: RegionValueBuilder): Unit = {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      rvb.addString(parseInfoString())
    }
  }

  def parseAddInfoDouble(rvb: RegionValueBuilder): Unit = {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      rvb.addDouble(VCFUtils.parseVcfDouble(parseInfoString()))
    }
  }

  def parseIntInInfoArray(): Int = {
    if (endInfoArrayElement())
      parseError("empty integer")
    var mul = 1
    if (line(pos) == '-') {
      mul = -1
      pos += 1
    }
    var v = 0
    while (!endInfoArrayElement()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v * mul
  }

  def parseStringInInfoArray(): String = {
    val start = pos
    while (!endInfoArrayElement())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseDoubleInInfoArray(): Double = VCFUtils.parseVcfDouble(parseStringInInfoArray())

  def parseInfoArrayIntElement(): Unit = {
    if (infoArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in INFO array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abi.addMissing()
      pos += 1 // dot
    } else
      abi += parseIntInInfoArray()
  }

  def parseInfoArrayStringElement(): Unit = {
    if (infoArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in INFO array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abs.addMissing()
      pos += 1 // dot
    } else
      abs += parseStringInInfoArray()
  }

  def parseInfoArrayDoubleElement(): Unit = {
    if (infoArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(
          "Missing value in INFO array. Use 'hl.import_vcf(..., array_elements_required=False)'."
        )
      abd.addMissing()
      pos += 1
    } else {
      abd += parseDoubleInInfoArray()
    }
  }

  def parseAddInfoArrayInt(rvb: RegionValueBuilder): Unit = {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      assert(abi.length == 0)
      parseInfoArrayIntElement()
      while (!endInfoField()) {
        pos += 1 // comma
        parseInfoArrayIntElement()
      }

      rvb.startArray(abi.length)
      var i = 0
      while (i < abi.length) {
        if (abi.isMissing(i))
          rvb.setMissing()
        else
          rvb.addInt(abi(i))
        i += 1
      }
      rvb.endArray()
      abi.clear()
    }
  }

  def parseAddInfoArrayString(rvb: RegionValueBuilder): Unit = {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      assert(abs.length == 0)
      parseInfoArrayStringElement()
      while (!endInfoField()) {
        pos += 1 // comma
        parseInfoArrayStringElement()
      }

      rvb.startArray(abs.length)
      var i = 0
      while (i < abs.length) {
        if (abs.isMissing(i))
          rvb.setMissing()
        else
          rvb.addString(abs(i))
        i += 1
      }
      rvb.endArray()
      abs.clear()
    }
  }

  def parseAddInfoArrayDouble(rvb: RegionValueBuilder): Unit = {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      assert(abd.length == 0)
      parseInfoArrayDoubleElement()
      while (!endInfoField()) {
        pos += 1 // comma
        parseInfoArrayDoubleElement()
      }

      rvb.startArray(abd.length)
      var i = 0
      while (i < abd.length) {
        if (abd.isMissing(i))
          rvb.setMissing()
        else
          rvb.addDouble(abd(i))
        i += 1
      }
      rvb.endArray()
      abd.clear()
    }
  }

  def parseAddInfoField(rvb: RegionValueBuilder, typ: Type): Unit = {
    val c = line(pos)
    if (c != ';' && c != '\t') {
      if (c != '=')
        parseError(s"invalid INFO key/value expression found '${line(pos)}' instead of '='")
      pos += 1 // equals
      typ match {
        case TInt32 => parseAddInfoInt(rvb)
        case TString => parseAddInfoString(rvb)
        case TFloat64 => parseAddInfoDouble(rvb)
        case TArray(TInt32) => parseAddInfoArrayInt(rvb)
        case TArray(TFloat64) => parseAddInfoArrayDouble(rvb)
        case TArray(TString) => parseAddInfoArrayString(rvb)
      }
    }
  }

  def addInfoField(key: String, rvb: RegionValueBuilder, c: ParseLineContext): Unit = {
    if (c.infoFields.containsKey(key)) {
      val idx = c.infoFields.get(key)
      rvb.setFieldIndex(idx)
      if (c.infoFlagFieldNames.contains(key)) {
        if (pos != line.length && line(pos) == '=') {
          pos += 1
          val s = parseInfoString()
          if (s != "0")
            rvb.addBoolean(true)
        } else
          rvb.addBoolean(true)
      } else {
        try
          parseAddInfoField(rvb, c.infoFieldTypes(idx))
        catch {
          case e: VCFParseError => parseError(s"error while parsing info field '$key': ${e.msg}")
        }
      }
    }
  }

  def parseAddInfo(rvb: RegionValueBuilder, c: ParseLineContext): Unit = {
    rvb.startStruct(init = true, setMissing = true)
    var i = 0
    while (i < c.infoFieldFlagIndices.length) {
      rvb.setFieldIndex(c.infoFieldFlagIndices(i))
      rvb.addBoolean(false)
      i += 1
    }

    // handle first key, which may be '.' for missing info
    var key = parseInfoKey()
    if (key == ".") {
      if (endField()) {
        rvb.setFieldIndex(c.infoFieldTypes.length)
        rvb.endStruct()
        return
      } else
        parseError(s"invalid INFO key $key")
    }

    addInfoField(key, rvb, c)
    skipInfoField()

    while (!endField()) {
      nextInfoField()
      key = parseInfoKey()
      if (key == ".") {
        parseError(s"invalid INFO key $key")
      }
      addInfoField(key, rvb, c)
      skipInfoField()
    }

    rvb.setFieldIndex(c.infoFieldTypes.size)
    rvb.endStruct()
  }
}

object FormatParser {
  def apply(gType: TStruct, format: String): FormatParser = {
    val formatFields = format.split(":")
    val formatFieldsSet = formatFields.toSet
    new FormatParser(
      gType,
      formatFields.map(f => gType.fieldIdx.getOrElse(f, -1)), // -1 means field has been pruned
      gType.fields.filter(f => !formatFieldsSet.contains(f.name)).map(_.index).toArray,
    )
  }
}

final class FormatParser(
  gType: TStruct,
  formatFieldGIndex: Array[Int],
  missingGIndices: Array[Int],
) {

  def parseAddField(l: VCFLine, rvb: RegionValueBuilder, i: Int): Unit = {
    // negative j values indicate field is pruned
    val j = formatFieldGIndex(i)
    if (j == -1)
      l.skipFormatField()
    else {
      rvb.setFieldIndex(j)
      gType.types(j) match {
        case TCall =>
          l.parseAddCall(rvb)
        case TInt32 =>
          l.parseAddFormatInt(rvb)
        case TFloat32 =>
          l.parseAddFormatFloat(rvb)
        case TFloat64 =>
          l.parseAddFormatDouble(rvb)
        case TString =>
          l.parseAddFormatString(rvb)
        case TArray(TInt32) =>
          l.parseAddFormatArrayInt(rvb)
        case TArray(TFloat32) =>
          l.parseAddFormatArrayFloat(rvb)
        case TArray(TFloat64) =>
          l.parseAddFormatArrayDouble(rvb)
        case TArray(TString) =>
          l.parseAddFormatArrayString(rvb)
      }
    }
  }

  def setMissing(rvb: RegionValueBuilder, i: Int): Unit = {
    val idx = formatFieldGIndex(i)
    if (idx >= 0) {
      rvb.setFieldIndex(idx)
      rvb.setMissing()
    }
  }

  def parse(l: VCFLine, rvb: RegionValueBuilder): Unit = {
    rvb.startStruct() // g

    // FIXME do in bulk, add setDefinedIndex
    var i = 0
    while (i < missingGIndices.length) {
      val j = missingGIndices(i)
      rvb.setFieldIndex(j)
      rvb.setMissing()
      i += 1
    }

    parseAddField(l, rvb, 0)
    var end = l.endField()
    i = 1
    while (i < formatFieldGIndex.length) {
      if (end)
        setMissing(rvb, i)
      else {
        l.nextFormatField()
        parseAddField(l, rvb, i)
        end = l.endField()
      }
      i += 1
    }

    // for error checking
    rvb.setFieldIndex(gType.size)

    rvb.endStruct() // g
  }
}

class ParseLineContext(
  val rowType: TStruct,
  val infoFlagFieldNames: java.util.HashSet[String],
  val nSamples: Int,
  val fileNum: Int,
  val entriesName: String,
) {
  val entryType: TStruct = rowType.selfField(entriesName) match {
    case Some(entriesArray) =>
      entriesArray.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
    case None => TStruct.empty
  }

  val infoSignature = rowType.selfField("info").map(_.typ.asInstanceOf[TStruct]).orNull
  val hasQual = rowType.hasField("qual")
  val hasFilters = rowType.hasField("filters")
  val hasEntryFields = entryType.size > 0

  val infoFields: java.util.HashMap[String, Int] =
    if (infoSignature != null) makeJavaMap(infoSignature.fieldIdx) else null

  val infoFieldTypes: Array[Type] = if (infoSignature != null) infoSignature.types else null

  val infoFieldFlagIndices: Array[Int] = if (infoSignature != null) {
    infoSignature.fields
      .iterator
      .filter(f => infoFlagFieldNames.contains(f.name))
      .map(_.index)
      .toArray
  } else
    null

  val formatParsers = new java.util.HashMap[String, FormatParser]()

  def getFormatParser(format: String): FormatParser = {
    if (formatParsers.containsKey(format))
      formatParsers.get(format)
    else {
      val fp = FormatParser(entryType, format)
      formatParsers.put(format, fp)
      fp
    }
  }
}

object LoadVCF {
  def warnDuplicates(ids: Array[String]): Unit = {
    val duplicates = ids.counter().filter(_._2 > 1)
    if (duplicates.nonEmpty) {
      warn(
        s"Found ${duplicates.size} duplicate ${plural(duplicates.size, "sample ID")}:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) =>
          s"""($count) "$id""""
        }.truncatable("\n  "),
      )
    }
  }

  def getEntryFloatType(entryFloatTypeName: String): TNumeric = {
    IRParser.parseType(entryFloatTypeName) match {
      case TFloat32 => TFloat32
      case TFloat64 => TFloat64
      case _ => fatal(
          s"""invalid floating point type:
        |  expected ${TFloat32._toPretty} or ${TFloat64._toPretty}, got $entryFloatTypeName"""
        )
    }
  }

  def headerNumberToString(line: VCFCompoundHeaderLine): String = line.getCountType match {
    case VCFHeaderLineCount.A => "A"
    case VCFHeaderLineCount.G => "G"
    case VCFHeaderLineCount.R => "R"
    case VCFHeaderLineCount.INTEGER => line.getCount.toString
    case VCFHeaderLineCount.UNBOUNDED => "."
  }

  def headerTypeToString(line: VCFCompoundHeaderLine): String = line.getType match {
    case VCFHeaderLineType.Integer => "Integer"
    case VCFHeaderLineType.Flag => "Flag"
    case VCFHeaderLineType.Float => "Float"
    case VCFHeaderLineType.Character => "Character"
    case VCFHeaderLineType.String => "String"
  }

  def headerField(
    line: VCFCompoundHeaderLine
  ): ((String, Type), (String, Map[String, String]), Boolean) = {
    val id = line.getID

    val baseType = line.getType match {
      case VCFHeaderLineType.Integer => TInt32
      case VCFHeaderLineType.Float => TFloat64
      case VCFHeaderLineType.String => TString
      case VCFHeaderLineType.Character => TString
      case VCFHeaderLineType.Flag => TBoolean
    }

    val attrs = Map(
      "Description" -> line.getDescription,
      "Number" -> headerNumberToString(line),
      "Type" -> headerTypeToString(line),
    )

    val isFlag = line.getType == VCFHeaderLineType.Flag

    if (
      line.isFixedCount &&
      (line.getCount == 1 ||
        (isFlag && line.getCount == 0))
    )
      ((id, baseType), (id, attrs), isFlag)
    else if (isFlag) {
      warn(
        s"invalid VCF header: at INFO field '$id' of type 'Flag', expected 'Number=0', got 'Number=${headerNumberToString(line)}''" +
          s"\n  Interpreting as 'Number=0' regardless."
      )
      ((id, baseType), (id, attrs), isFlag)
    } else if (baseType.isInstanceOf[PCall])
      fatal("fields in 'call_fields' must have 'Number' equal to 1.")
    else
      ((id, TArray(baseType)), (id, attrs), isFlag)
  }

  def headerSignature[T <: VCFCompoundHeaderLine](
    lines: java.util.Collection[T]
  ): (Array[(String, Type)], VCFAttributes, Set[String]) = {
    val (fields, attrs, flags) = lines.asScala
      .map(line => headerField(line))
      .unzip3

    val flagFieldNames = fields.zip(flags)
      .flatMap { case ((f, _), isFlag) => if (isFlag) Some(f) else None }
      .toSet

    (fields.toArray, attrs.toMap, flagFieldNames)
  }

  def parseHeader(
    lines: Array[String]
  ): VCFHeaderInfo = {
    val codec = new htsjdk.variant.vcf.VCFCodec()
    // Disable "repairing" of headers by htsjdk according to the VCF standard.
    codec.disableOnTheFlyModifications()
    val header = codec.readHeader(new BufferedLineIterator(lines.iterator.buffered))
      .getHeaderValue
      .asInstanceOf[htsjdk.variant.vcf.VCFHeader]

    val filterAttrs: VCFAttributes = header
      .getFilterLines
      .asScala
      .toList
      // (ID, description)
      .map(line => (line.getID, Map("Description" -> line.getDescription)))
      .toMap

    val infoHeader = header.getInfoHeaderLines
    val (infoSignature, infoAttrs, infoFlagFields) = headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val (gSignature, formatAttrs, _) = headerSignature(formatHeader)

    val headerLine = lines.last
    if (!(headerLine(0) == '#' && headerLine(1) != '#'))
      fatal(
        s"""corrupt VCF: expected final header line of format '#CHROM\tPOS\tID...'
           |  found: @1""".stripMargin,
        headerLine,
      )

    val sampleIds: Array[String] = headerLine.split("\t").drop(9)

    VCFHeaderInfo(
      sampleIds,
      infoSignature,
      gSignature,
      filterAttrs,
      infoAttrs,
      formatAttrs,
      infoFlagFields)
  }

  def getHeaderLines[T](
    fs: FS,
    file: String,
    filterAndReplace: TextInputFilterAndReplace,
  ): Array[String] = fs.readLines(file, filterAndReplace) { lines =>
    lines
      .takeWhile(line => line.value(0) == '#')
      .map(_.value)
      .toArray
  }

  def getVCFHeaderInfo(fs: FS, file: String, filter: String, find: String, replace: String)
    : VCFHeaderInfo =
    parseHeader(getHeaderLines(
      fs,
      file,
      TextInputFilterAndReplace(Option(filter), Option(find), Option(replace)),
    ))

  def parseLine(
    rg: Option[ReferenceGenome],
    contigRecoding: Map[String, String],
    skipInvalidLoci: Boolean,
    rowPType: PStruct,
    rvb: RegionValueBuilder,
    parseLineContext: ParseLineContext,
    vcfLine: VCFLine,
    entriesFieldName: String = LowerMatrixIR.entriesFieldName,
    uidFieldName: String = MatrixReader.rowUIDFieldName,
  ): Boolean = {
    val hasLocus = rowPType.hasField("locus")
    val hasAlleles = rowPType.hasField("alleles")
    val hasRSID = rowPType.hasField("rsid")
    val hasEntries = rowPType.hasField(entriesFieldName)
    val hasRowUID = rowPType.hasField(uidFieldName)

    rvb.start(rowPType)
    rvb.startStruct()
    val present = vcfLine.parseAddVariant(rvb, rg, contigRecoding, hasLocus, hasAlleles, hasRSID,
      skipInvalidLoci)
    if (!present)
      return present

    parseLineInner(parseLineContext, vcfLine, rvb, !hasEntries)

    if (hasRowUID) {
      rvb.startTuple()
      rvb.addLong(vcfLine.fileNum)
      rvb.addLong(vcfLine.fileOffset)
      rvb.endTuple()
    }

    true
  }

  // parses the Variant (key), and ID if necessary, leaves the rest to f
  def parseLines(
    makeContext: () => ParseLineContext
  )(
    f: (ParseLineContext, VCFLine, RegionValueBuilder) => Unit
  )(
    lines: ContextRDD[WithContext[String]],
    rowPType: PStruct,
    rgBc: Option[BroadcastValue[ReferenceGenome]],
    contigRecoding: Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean,
  ): ContextRDD[Long] = {
    val hasRSID = rowPType.hasField("rsid")
    lines.cmapPartitions { (ctx, it) =>
      new Iterator[Long] {
        val rvb = ctx.rvb
        var ptr: Long = 0

        val context: ParseLineContext = makeContext()

        var present: Boolean = false

        val abs = new MissingArrayBuilder[String]
        val abi = new MissingArrayBuilder[Int]
        val abf = new MissingArrayBuilder[Float]
        val abd = new MissingArrayBuilder[Double]

        def hasNext: Boolean = {
          while (!present && it.hasNext) {
            val lwc = it.next()
            val line = lwc.value
            try {
              val vcfLine = new VCFLine(
                line,
                context.fileNum,
                lwc.source.position.get,
                arrayElementsRequired,
                abs,
                abi,
                abf,
                abd,
              )
              rvb.start(rowPType)
              rvb.startStruct()
              present = vcfLine.parseAddVariant(
                rvb,
                rgBc.map(_.value),
                contigRecoding,
                hasRSID,
                true,
                true,
                skipInvalidLoci,
              )
              if (present) {
                f(context, vcfLine, rvb)

                rvb.endStruct()
                ptr = rvb.end()
              } else
                rvb.clear()
            } catch {
              case e: VCFParseError =>
                val pos = e.pos
                val source = lwc.source

                val excerptStart = math.max(0, pos - 36)
                val excerptEnd = math.min(line.length, pos + 36)
                val excerpt = line.substring(excerptStart, excerptEnd)
                  .map(c => if (c == '\t') ' ' else c)

                val prefix = if (excerptStart > 0) "... " else ""
                val suffix = if (excerptEnd < line.length) " ..." else ""

                var caretPad = prefix.length + pos - excerptStart
                var pad = " " * caretPad

                fatal(
                  s"${source.locationString(pos)}: ${e.msg}\n$prefix$excerpt$suffix\n$pad^\noffending line: @1\nsee the Hail log for the full offending line",
                  line,
                  e,
                )
              case e: Throwable =>
                lwc.source.wrapException(e)
            }
          }
          present
        }

        def next(): Long = {
          // call hasNext to advance if necessary
          if (!hasNext)
            throw new java.util.NoSuchElementException()
          present = false
          ptr
        }
      }
    }
  }

  def parseHeaderMetadata(
    fs: FS,
    callFields: Set[String],
    entryFloatType: TNumeric,
    headerFile: String,
  ): VCFMetadata = {
    val headerLines = getHeaderLines(fs, headerFile, TextInputFilterAndReplace())
    val header = parseHeader(headerLines)

    Map("filter" -> header.filtersAttrs, "info" -> header.infoAttrs, "format" -> header.formatAttrs)
  }

  def parseLineInner(
    c: ParseLineContext,
    l: VCFLine,
    rvb: RegionValueBuilder,
    dropSamples: Boolean = false,
  ): Unit = {
    // QUAL
    if (c.hasQual) {
      val qstr = l.parseString()
      if (qstr == ".")
        rvb.addDouble(-10.0)
      else
        rvb.addDouble(qstr.toDouble)
    } else
      l.skipField()
    l.nextField()

    // filters
    if (c.hasFilters) {
      if (l.parseFilters()) {
        rvb.startArray(l.abs.length)
        var i = 0
        while (i < l.abs.length) {
          rvb.addString(l.abs(i))
          i += 1
        }
        rvb.endArray()
        l.abs.clear()
      } else
        rvb.setMissing()
    } else
      l.skipField()
    l.nextField()

    // info
    if (c.infoSignature != null)
      l.parseAddInfo(rvb, c)
    else
      l.skipField()

    if (!dropSamples) {
      rvb.startArray(c.nSamples) // gs

      if (c.nSamples > 0) {
        if (!c.hasEntryFields) {
          var i = 0
          while (i < c.nSamples) {
            rvb.startStruct()
            rvb.endStruct()
            i += 1
          }
        } else {
          l.nextField() // move past INFO
          val format = l.parseString()
          val fp = c.getFormatParser(format)

          var i = 0
          while (i < c.nSamples) {
            l.nextField()
            fp.parse(l, rvb)
            i += 1
          }
        }
      }
      rvb.endArray()
    }
  }
}

case class PartitionedVCFPartition(index: Int, chrom: String, start: Int, end: Int)
    extends Partition

class PartitionedVCFRDD(
  fsBc: BroadcastValue[FS],
  file: String,
  @(transient @param) reverseContigMapping: Map[String, String],
  @(transient @param) _partitions: Array[Partition],
) extends RDD[WithContext[String]](SparkBackend.sparkContext("PartitionedVCFRDD"), Seq()) {

  val contigRemappingBc =
    if (reverseContigMapping.size != 0) sparkContext.broadcast(reverseContigMapping) else null

  protected def getPartitions: Array[Partition] = _partitions

  def compute(split: Partition, context: TaskContext): Iterator[WithContext[String]] = {
    val p = split.asInstanceOf[PartitionedVCFPartition]

    val chromToQuery = if (contigRemappingBc != null)
      contigRemappingBc.value.getOrElse(p.chrom, p.chrom)
    else p.chrom

    val reg = {
      val r = new TabixReader(file, fsBc.value)
      val tid = r.chr2tid(chromToQuery)
      r.queryPairs(tid, p.start - 1, p.end)
    }
    if (reg.isEmpty)
      return Iterator.empty

    val lines = new TabixLineIterator(fsBc.value, file, reg)

    // clean up
    val context = TaskContext.get
    context.addTaskCompletionListener[Unit]((context: TaskContext) => lines.close())

    val it: Iterator[WithContext[String]] = new Iterator[WithContext[String]] {
      private var l = lines.next()
      private var curIdx: Long = lines.getCurIdx()

      def hasNext: Boolean = l != null

      def next(): WithContext[String] = {
        assert(l != null)
        val n = l
        l = lines.next()
        curIdx = lines.getCurIdx()
        if (l == null)
          lines.close()
        WithContext(n, Context(n, file, Some(curIdx.toInt)))
      }
    }

    it.filter { l =>
      val t1 = l.value.indexOf('\t')
      val t2 = l.value.indexOf('\t', t1 + 1)

      val chrom = l.value.substring(0, t1)
      val pos = l.value.substring(t1 + 1, t2).toInt

      if (chrom != chromToQuery) {
        throw new RuntimeException(s"bad chromosome! $chromToQuery, $l")
      }
      p.start <= pos && pos <= p.end
    }
  }
}

object MatrixVCFReader {
  def apply(
    ctx: ExecuteContext,
    files: Seq[String],
    callFields: Set[String],
    entryFloatTypeName: String,
    headerFile: Option[String],
    sampleIDs: Option[Seq[String]],
    nPartitions: Option[Int],
    blockSizeInMB: Option[Int],
    minPartitions: Option[Int],
    rg: Option[String],
    contigRecoding: Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean,
    gzAsBGZ: Boolean,
    forceGZ: Boolean,
    filterAndReplace: TextInputFilterAndReplace,
    partitionsJSON: Option[String],
    partitionsTypeStr: Option[String],
  ): MatrixVCFReader =
    MatrixVCFReader(
      ctx,
      MatrixVCFReaderParameters(
        files, callFields, entryFloatTypeName, headerFile, sampleIDs, nPartitions, blockSizeInMB,
        minPartitions, rg,
        contigRecoding, arrayElementsRequired, skipInvalidLoci, gzAsBGZ, forceGZ, filterAndReplace,
        partitionsJSON, partitionsTypeStr),
    )

  def apply(ctx: ExecuteContext, params: MatrixVCFReaderParameters): MatrixVCFReader = {
    val backend = ctx.backend
    val fs = ctx.fs

    val referenceGenome = params.rg.map(ctx.getReference)

    referenceGenome.foreach(_.validateContigRemap(params.contigRecoding))

    val fileListEntries = fs.globAll(params.files)
    fileListEntries.map(_.getPath).foreach { path =>
      if (!(path.endsWith(".vcf") || path.endsWith(".vcf.bgz") || path.endsWith(".vcf.gz")))
        warn(s"expected input file '$path' to end in .vcf[.bgz, .gz]")
    }
    checkGzipOfGlobbedFiles(params.files, fileListEntries, params.forceGZ, params.gzAsBGZ)

    val entryFloatType = LoadVCF.getEntryFloatType(params.entryFloatTypeName)

    val headerLines1 = getHeaderLines(
      fs,
      params.headerFile.getOrElse(fileListEntries.head.getPath),
      params.filterAndReplace,
    )
    val header1 = parseHeader(headerLines1)

    if (fileListEntries.length > 1) {
      if (params.headerFile.isEmpty) {
        val header1Bc = backend.broadcast(header1)

        val localCallFields = params.callFields
        val localFloatType = entryFloatType
        val files = fileListEntries.map(_.getPath)
        val localArrayElementsRequired = params.arrayElementsRequired
        val localFilterAndReplace = params.filterAndReplace

        val fsConfigBC = backend.broadcast(fs.getConfiguration())
        backend.parallelizeAndComputeWithIndex(
          ctx.backendContext,
          fs,
          files.tail.map(_.getBytes),
          "load_vcf_parse_header",
          None,
        ) { (bytes, htc, _, fs) =>
          val fsConfig = fsConfigBC.value
          fs.setConfiguration(fsConfig)
          val file = new String(bytes)

          val hd = parseHeader(getHeaderLines(fs, file, localFilterAndReplace))
          val hd1 = header1Bc.value

          if (params.sampleIDs.isEmpty && hd1.sampleIds.length != hd.sampleIds.length) {
            fatal(
              s"""invalid sample IDs: expected same number of samples for all inputs.
                 | ${files(0)} has ${hd1.sampleIds.length} ids and
                 | $file has ${hd.sampleIds.length} ids.
         """.stripMargin
            )
          }

          if (params.sampleIDs.isEmpty) {
            hd1.sampleIds.iterator.zipAll(hd.sampleIds.iterator, None, None)
              .zipWithIndex.foreach { case ((s1, s2), i) =>
                if (s1 != s2) {
                  fatal(
                    s"""invalid sample IDs: expected sample ids to be identical for all inputs. Found different sample IDs at position $i.
                       |    ${files(0)}: $s1
                       |    $file: $s2""".stripMargin
                  )
                }
              }
          }

          if (!hd.formatCompatible(hd1))
            fatal(
              s"""invalid genotype signature: expected signatures to be identical for all inputs.
                 |   ${files(0)}: ${hd1.genotypeSignature.toString}
                 |   $file: ${hd.genotypeSignature.toString}""".stripMargin
            )

          if (!hd.infoCompatible(hd1))
            fatal(
              s"""invalid variant annotation signature: expected signatures to be identical for all inputs. Check that all files have same INFO fields.
                 |   ${files(0)}: ${hd1.infoSignature.toString}
                 |   $file: ${hd.infoSignature.toString}""".stripMargin
            )

          bytes
        }
      }
    }

    val sampleIDs = params.sampleIDs.map(_.toArray).getOrElse(header1.sampleIds)

    LoadVCF.warnDuplicates(sampleIDs)

    new MatrixVCFReader(
      params.copy(files = fileListEntries.map(_.getPath)),
      fileListEntries,
      referenceGenome,
      header1,
    )
  }

  def fromJValue(ctx: ExecuteContext, jv: JValue): MatrixVCFReader = {
    implicit val formats: Formats = DefaultFormats
    val params = jv.extract[MatrixVCFReaderParameters]

    MatrixVCFReader(ctx, params)
  }
}

case class MatrixVCFReaderParameters(
  files: Seq[String],
  callFields: Set[String],
  entryFloatTypeName: String,
  headerFile: Option[String],
  sampleIDs: Option[Seq[String]],
  nPartitions: Option[Int],
  blockSizeInMB: Option[Int],
  minPartitions: Option[Int],
  rg: Option[String],
  contigRecoding: Map[String, String],
  arrayElementsRequired: Boolean,
  skipInvalidLoci: Boolean,
  gzAsBGZ: Boolean,
  forceGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  partitionsJSON: Option[String],
  partitionsTypeStr: Option[String],
) {
  require(
    partitionsJSON.isEmpty == partitionsTypeStr.isEmpty,
    "partitions and type must either both be defined or undefined",
  )
}

class MatrixVCFReader(
  val params: MatrixVCFReaderParameters,
  fileListEntries: IndexedSeq[FileListEntry],
  referenceGenome: Option[ReferenceGenome],
  header: VCFHeaderInfo,
) extends MatrixHybridReader {
  require(
    params.partitionsJSON.isEmpty || fileListEntries.length == 1,
    "reading with partitions can currently only read a single path",
  )

  val sampleIDs = params.sampleIDs.map(_.toArray).getOrElse(header.sampleIds)

  LoadVCF.warnDuplicates(sampleIDs)

  def rowUIDType = TTuple(TInt64, TInt64)
  def colUIDType = TInt64

  val (infoPType, rowValuePType, formatPType) = header.getPTypes(
    params.arrayElementsRequired,
    IRParser.parseType(params.entryFloatTypeName),
    params.callFields,
  )

  def fullMatrixTypeWithoutUIDs: MatrixType = MatrixType(
    globalType = TStruct.empty,
    colType = TStruct("s" -> TString),
    colKey = Array("s"),
    rowType = TStruct(
      Array(
        "locus" -> TLocus.schemaFromRG(referenceGenome.map(_.name)),
        "alleles" -> TArray(TString),
      )
        ++ rowValuePType.fields.map(f => f.name -> f.typ.virtualType): _*
    ),
    rowKey = Array("locus", "alleles"),
    // rowKey = Array.empty[String],
    entryType = formatPType.virtualType,
  )

  val fullRVDType = RVDType(
    PCanonicalStruct(
      true,
      FastSeq(
        "locus" -> PCanonicalLocus.schemaFromRG(referenceGenome.map(_.name), true),
        "alleles" -> PCanonicalArray(PCanonicalString(true), true),
      )
        ++ rowValuePType.fields.map(f => f.name -> f.typ)
        ++ FastSeq(
          LowerMatrixIR.entriesFieldName -> PCanonicalArray(formatPType, true),
          rowUIDFieldName -> PCanonicalTuple(true, PInt64Required, PInt64Required),
        ): _*
    ),
    fullType.key,
  )

  def pathsUsed: Seq[String] = params.files

  def nCols: Int = sampleIDs.length

  val columnCount: Option[Int] = Some(nCols)

  val partitionCounts: Option[IndexedSeq[Long]] = None

  def partitioner(sm: HailStateManager): Option[RVDPartitioner] =
    params.partitionsJSON.map { partitionsJSON =>
      val indexedPartitionsType = IRParser.parseType(params.partitionsTypeStr.get)
      val jv = JsonMethods.parse(partitionsJSON)
      val rangeBounds = JSONAnnotationImpex.importAnnotation(jv, indexedPartitionsType)
        .asInstanceOf[IndexedSeq[Interval]]

      rangeBounds.foreach { bound =>
        if (!(bound.includesStart && bound.includesEnd))
          fatal("range bounds must be inclusive")

        val start = bound.start.asInstanceOf[Row].getAs[Locus](0)
        val end = bound.end.asInstanceOf[Row].getAs[Locus](0)
        if (start.contig != end.contig)
          fatal(
            s"partition spec must not cross contig boundaries, start: ${start.contig} | end: ${end.contig}"
          )
      }
      new RVDPartitioner(
        sm,
        Array("locus"),
        fullType.keyType,
        rangeBounds,
      )
    }

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(tcoerce[PStruct](fullRVDType.rowType.subsetTo(requestedType.rowType)))

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(PType.canonical(requestedType.globalType))

  def executeGeneric(ctx: ExecuteContext, dropRows: Boolean = false): GenericTableValue = {
    val fs = ctx.fs
    val sm = ctx.stateManager

    val rgBc = referenceGenome.map(_.broadcast)
    val localArrayElementsRequired = params.arrayElementsRequired
    val localContigRecoding = params.contigRecoding
    val localSkipInvalidLoci = params.skipInvalidLoci
    val localInfoFlagFieldNames = header.infoFlagFields
    val localNSamples = nCols
    val localFilterAndReplace = params.filterAndReplace

    val part = partitioner(ctx.stateManager)
    val lines = part match {
      case Some(partitioner) =>
        GenericLines.readTabix(
          fs,
          fileListEntries(0).getPath,
          localContigRecoding,
          partitioner.rangeBounds,
        )
      case None =>
        GenericLines.read(
          fs,
          fileListEntries,
          params.nPartitions,
          params.blockSizeInMB,
          params.minPartitions,
          params.gzAsBGZ,
          params.forceGZ,
        )
    }

    val globals = Row(sampleIDs.zipWithIndex.map { case (s, i) => Row(s, i.toLong) }.toFastSeq)

    val fullRowPType: PType = fullRVDType.rowType

    val bodyPType =
      (requestedRowType: TStruct) => fullRowPType.subsetTo(requestedRowType).asInstanceOf[PStruct]

    val linesBody = if (dropRows) { (_: FS, _: Any) => CloseableIterator.empty[GenericLine] }
    else
      lines.body
    val body = { (requestedType: TStruct) =>
      val requestedPType = bodyPType(requestedType)

      { (region: Region, theHailClassLoader: HailClassLoader, fs: FS, context: Any) =>
        val fileNum = context.asInstanceOf[Row].getInt(1)
        val parseLineContext = new ParseLineContext(
          requestedType,
          makeJavaSet(localInfoFlagFieldNames),
          localNSamples,
          fileNum,
          LowerMatrixIR.entriesFieldName,
        )

        val rvb = new RegionValueBuilder(sm, region)

        val abs = new MissingArrayBuilder[String]
        val abi = new MissingArrayBuilder[Int]
        val abf = new MissingArrayBuilder[Float]
        val abd = new MissingArrayBuilder[Double]

        val transformer = localFilterAndReplace.transformer()

        linesBody(fs, context)
          .filter { line =>
            val text = line.toString
            val newText = transformer(text)
            if (newText != null) {
              rvb.clear()
              try {
                val vcfLine = new VCFLine(
                  newText,
                  line.fileNum,
                  line.offset,
                  localArrayElementsRequired,
                  abs,
                  abi,
                  abf,
                  abd,
                )
                LoadVCF.parseLine(
                  rgBc.map(_.value),
                  localContigRecoding,
                  localSkipInvalidLoci,
                  requestedPType,
                  rvb,
                  parseLineContext,
                  vcfLine,
                )
              } catch {
                case e: Exception =>
                  fatal(
                    s"${line.file}:offset ${line.offset}: error while parsing line\n" +
                      s"$newText\n",
                    e,
                  )
              }
            } else
              false
          }.map(_ => rvb.result().offset)
      }
    }

    new GenericTableValue(
      fullType,
      rowUIDFieldName,
      part,
      { (requestedGlobalsType: Type) =>
        val subset = fullType.globalType.valueSubsetter(requestedGlobalsType)
        subset(globals).asInstanceOf[Row]
      },
      lines.contextType.asInstanceOf[TStruct],
      lines.contexts,
      bodyPType,
      body,
    )
  }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val globals = Row(sampleIDs.zipWithIndex.map(t => Row(t._1, t._2.toLong)).toFastSeq)
    Literal.coerce(
      requestedGlobalsType,
      fullType.globalType.valueSubsetter(requestedGlobalsType)
        .apply(globals),
    )
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType, "VCF", params)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixVCFReader")
  }

  def renderShort(): String = defaultRender()

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixVCFReader => params == that.params
    case _ => false
  }
}

case class GVCFPartitionReader(
  header: VCFHeaderInfo,
  callFields: Set[String],
  entryFloatType: Type,
  arrayElementsRequired: Boolean,
  rg: Option[String],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  entriesFieldName: String,
  uidFieldName: String,
) extends PartitionReader {

  lazy val contextType: TStruct = TStruct(
    "fileNum" -> TInt32,
    "path" -> TString,
    "contig" -> TString,
    "start" -> TInt32,
    "end" -> TInt32,
  )

  lazy val (infoType, rowValueType, entryType) =
    header.getPTypes(arrayElementsRequired, entryFloatType, callFields)

  lazy val fullRowPType: PCanonicalStruct = PCanonicalStruct(
    true,
    FastSeq(
      ("locus", PCanonicalLocus.schemaFromRG(rg, true)),
      ("alleles", PCanonicalArray(PCanonicalString(true), true)),
    )
      ++ rowValueType.fields.map(f => (f.name, f.typ))
      ++ Array(
        entriesFieldName -> PCanonicalArray(entryType, true),
        uidFieldName -> PCanonicalTuple(true, PInt64Required, PInt64Required),
      ): _*
  )

  lazy val fullRowType: TStruct = fullRowPType.virtualType

  def rowRequiredness(requestedType: TStruct): RStruct =
    VirtualTypeWithReq(tcoerce[PStruct](fullRowPType.subsetTo(requestedType))).r.asInstanceOf[
      RStruct
    ]

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(this, "MatrixVCFReader")
  }

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
  ): IEmitCode = {
    context.toI(cb).map(cb) { case ctxValue: SBaseStructValue =>
      val fileNum = cb.memoizeField(ctxValue.loadField(cb, "fileNum").get(cb).asInt32.value)
      val filePath = cb.memoizeField(ctxValue.loadField(cb, "path").get(cb).asString.loadString(cb))
      val contig = cb.memoizeField(ctxValue.loadField(cb, "contig").get(cb).asString.loadString(cb))
      val start = cb.memoizeField(ctxValue.loadField(cb, "start").get(cb).asInt32.value)
      val end = cb.memoizeField(ctxValue.loadField(cb, "end").get(cb).asInt32.value)

      val requestedPType = fullRowPType.subsetTo(requestedType).asInstanceOf[PStruct]
      val eltRegion = mb.genFieldThisRef[Region]("gvcf_elt_region")
      val iter = mb.genFieldThisRef[TabixReadVCFIterator]("gvcf_iter")
      val currentElt = mb.genFieldThisRef[Long]("curr_elt")

      SStreamValue(new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
          cb.assign(
            iter,
            Code.newInstance[TabixReadVCFIterator](
              Array[Class[_]](
                classOf[FS],
                classOf[String],
                classOf[Map[String, String]],
                classOf[Int],
                classOf[String],
                classOf[Int],
                classOf[Int],
                classOf[HailStateManager],
                classOf[Region],
                classOf[Region],
                classOf[PStruct],
                classOf[TextInputFilterAndReplace],
                classOf[Set[String]],
                classOf[Int],
                classOf[ReferenceGenome],
                classOf[Boolean],
                classOf[Boolean],
                classOf[String],
                classOf[String],
              ),
              Array[Code[_]](
                mb.getFS,
                filePath,
                mb.getObject(contigRecoding),
                fileNum,
                contig,
                start,
                end,
                cb.emb.getObject(cb.emb.ecb.ctx.stateManager),
                outerRegion,
                eltRegion,
                mb.getPType(requestedPType),
                mb.getObject(filterAndReplace),
                mb.getObject(header.infoFlagFields),
                const(header.sampleIds.length),
                rg.map(mb.getReferenceGenome).getOrElse(Code._null[ReferenceGenome]),
                const(arrayElementsRequired),
                const(skipInvalidLoci),
                const(entriesFieldName),
                const(uidFieldName),
              ),
            ),
          )
        }

        override val elementRegion: Settable[Region] = eltRegion
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.assign(currentElt, iter.invoke[Region, Long]("next", eltRegion))
          cb.if_(currentElt ceq 0L, cb.goto(LendOfStream), cb.goto(LproduceElementDone))
        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb =>
          IEmitCode.present(cb, requestedPType.loadCheapSCode(cb, currentElt))
        )
        override def close(cb: EmitCodeBuilder): Unit = {
          cb += iter.invoke[Unit]("close")
          cb.assign(iter, Code._null)
        }
      })
    }

  }
}

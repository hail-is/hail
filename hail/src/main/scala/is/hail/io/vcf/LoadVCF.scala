package is.hail.io.vcf

import htsjdk.variant.vcf._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.BroadcastValue
import is.hail.backend.spark.SparkBackend
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.ir.{ExecuteContext, GenericLine, GenericLines, GenericTableValue, IRParser, LowerMatrixIR, LoweredTableReader, MatrixHybridReader, MatrixIR, MatrixLiteral, PruneDeadFields, TableRead, TableValue}
import is.hail.types._
import is.hail.types.physical.{PBoolean, PCall, PCanonicalArray, PCanonicalCall, PCanonicalLocus, PCanonicalSet, PCanonicalString, PCanonicalStruct, PField, PFloat64, PInt32, PStruct, PType}
import is.hail.types.virtual._
import is.hail.io.fs.{FS, FileStatus}
import is.hail.io.tabix._
import is.hail.io.vcf.LoadVCF.{getHeaderLines, parseHeader, parseLines}
import is.hail.io.{VCFAttributes, VCFMetadata}
import is.hail.rvd.{RVD, RVDCoercer, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.json4s.JsonAST.{JInt, JObject}
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.annotation.meta.param
import scala.annotation.switch
import scala.collection.JavaConverters._
import scala.language.implicitConversions
import org.json4s.{DefaultFormats, Extraction, Formats, JValue}

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() {
    throw new UnsupportedOperationException
  }
}

case class VCFHeaderInfo(sampleIds: Array[String], infoSignature: PStruct, vaSignature: PStruct, genotypeSignature: PStruct,
  filtersAttrs: VCFAttributes, infoAttrs: VCFAttributes, formatAttrs: VCFAttributes, infoFlagFields: Set[String])

class VCFParseError(val msg: String, val pos: Int) extends RuntimeException(msg)


final class VCFLine(val line: String, arrayElementsRequired: Boolean,
  val abs: MissingArrayBuilder[String],
  val abi: MissingArrayBuilder[Int],
  val abf: MissingArrayBuilder[Float],
  val abd: MissingArrayBuilder[Double]) {
  var pos: Int = 0

  def parseError(msg: String): Unit = throw new VCFParseError(msg, pos)

  def numericValue(c: Char): Int = {
    if (c < '0' || c > '9')
      parseError(s"invalid character '${StringEscapeUtils.escapeString(c.toString)}' in integer literal")
    c - '0'
  }

  // field contexts: field, array field, format field, call field, format array field, filter array field

  def endField(p: Int): Boolean = {
    p == line.length || line(p) == '\t'
  }

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

  def skipInfoField(): Unit = {
    while (!endInfoField())
      pos += 1
  }

  def skipFormatField(): Unit = {
    while (!endFormatField())
      pos += 1
  }

  def fieldMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endField(pos + 1)
  }

  def arrayFieldMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endArrayElement(pos + 1)
  }

  def infoFieldMissing(): Boolean = {
    pos < line.length &&
      (line(pos) == '.' &&
       endInfoField(pos + 1) ||
       endInfoField(pos))
  }

  def formatFieldMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endFormatField(pos + 1)
  }

  def callFieldMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endCallField(pos + 1)
  }

  def infoArrayElementMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endInfoArrayElement(pos + 1)
  }

  def formatArrayElementMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endFormatArrayElement(pos + 1)
  }

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

  def skipField(): Unit = {
    while (!endField())
      pos += 1
  }

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
    skipInvalidLoci: Boolean): Boolean = {
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
        case None => { // Without a reference genome, we use a struct of two fields rather than a PLocus
          rvb.startStruct() // pk: Locus
          rvb.addString(recodedContig)
          rvb.addInt(start)
          rvb.endStruct()
        }
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

  def parseAddCall(rvb: RegionValueBuilder) {
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

  def parseAddFormatInt(rvb: RegionValueBuilder) {
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

  def parseAddFormatString(rvb: RegionValueBuilder) {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else
      rvb.addString(parseFormatString())
  }

  def parseFormatFloat(): Float = {
    val s = parseFormatString()
    s.toFloat
  }

  def parseAddFormatFloat(rvb: RegionValueBuilder) {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      rvb.addFloat(parseFormatFloat())
    }
  }

  def parseFormatDouble(): Double = {
    val s = parseFormatString()
    s.toDouble
  }

  def parseAddFormatDouble(rvb: RegionValueBuilder) {
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

  def parseArrayElement[T](ab: MissingArrayBuilder[T], eltParser: () => T) {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      ab.addMissing()
      pos += 1
    } else {
      ab += eltParser()
    }
  }

  def parseIntArrayElement() {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      abi.addMissing()
      pos += 1
    } else {
      abi += parseIntInFormatArray()
    }
  }

  def parseFloatArrayElement() {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      abf.addMissing()
      pos += 1
    } else {
      abf += parseFloatInFormatArray()
    }
  }

  def parseDoubleArrayElement() {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      abd.addMissing()
      pos += 1
    } else {
      abd += parseDoubleInFormatArray()
    }
  }

  def parseStringArrayElement() {
    if (formatArrayElementMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      abs.addMissing()
      pos += 1
    } else {
      abs += parseStringInFormatArray()
    }
  }

  def parseAddFormatArrayInt(rvb: RegionValueBuilder) {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abi.length == 0)

      parseIntArrayElement()

      while (!endFormatField()) {
        pos += 1 // comma
        parseIntArrayElement()
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

  def parseAddFormatArrayString(rvb: RegionValueBuilder) {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abs.length == 0)

      parseStringArrayElement()
      while (!endFormatField()) {
        pos += 1 // comma
        parseStringArrayElement()
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

  def parseAddFormatArrayFloat(rvb: RegionValueBuilder) {
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

  def parseAddFormatArrayDouble(rvb: RegionValueBuilder) {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abd.length == 0)

      parseDoubleArrayElement()
      while (!endFormatField()) {
        pos += 1 // comma
        parseDoubleArrayElement()
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

  // cdv: keep backwards compatibility with the old parser
  def infoToDouble(s: String): Double = {
    s match {
      case "nan" => Double.NaN
      case "-nan" => Double.NaN
      case "inf" => Double.PositiveInfinity
      case "-inf" => Double.NegativeInfinity
      case _ => s.toDouble
    }
  }

  def parseAddInfoInt(rvb: RegionValueBuilder) {
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

  def parseAddInfoString(rvb: RegionValueBuilder) {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      rvb.addString(parseInfoString())
    }
  }

  def parseAddInfoDouble(rvb: RegionValueBuilder) {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      rvb.addDouble(infoToDouble(parseInfoString()))
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

  def parseDoubleInInfoArray(): Double = infoToDouble(parseStringInInfoArray())

  def parseIntInfoArrayElement() {
    if (infoArrayElementMissing()) {
      abi.addMissing()
      pos += 1  // dot
    } else
      abi += parseIntInInfoArray()
  }

  def parseStringInfoArrayElement() {
    if (infoArrayElementMissing()) {
      abs.addMissing()
      pos += 1  // dot
    } else
      abs += parseStringInInfoArray()
  }

  def parseDoubleInfoArrayElement() {
    if (infoArrayElementMissing()) {
      abd.addMissing()
      pos += 1
    } else {
      abd += parseDoubleInInfoArray()
    }
  }

  def parseAddInfoArrayInt(rvb: RegionValueBuilder) {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      assert(abi.length == 0)
      parseIntInfoArrayElement()
      while (!endInfoField()) {
        pos += 1  // comma
        parseIntInfoArrayElement()
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

  def parseAddInfoArrayString(rvb: RegionValueBuilder) {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      assert(abs.length == 0)
      parseStringInfoArrayElement()
      while (!endInfoField()) {
        pos += 1  // comma
        parseStringInfoArrayElement()
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

  def parseAddInfoArrayDouble(rvb: RegionValueBuilder) {
    if (!infoFieldMissing()) {
      rvb.setPresent()
      assert(abd.length == 0)
      parseDoubleInfoArrayElement()
      while (!endInfoField()) {
        pos += 1  // comma
        parseDoubleInfoArrayElement()
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

  def parseAddInfoField(rvb: RegionValueBuilder, typ: Type) {
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
        try {
          parseAddInfoField(rvb, c.infoFieldTypes(idx))
        } catch {
          case e: VCFParseError => parseError(s"error while parsing info field '$key': ${ e.msg }")
        }
      }
    }
  }

  def parseAddInfo(rvb: RegionValueBuilder, c: ParseLineContext) {
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
      gType.fields.filter(f => !formatFieldsSet.contains(f.name)).map(_.index).toArray)
  }
}

final class FormatParser(
  gType: TStruct,
  formatFieldGIndex: Array[Int],
  missingGIndices: Array[Int]) {

  def parseAddField(l: VCFLine, rvb: RegionValueBuilder, i: Int) {
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

  def setMissing(rvb: RegionValueBuilder, i: Int) {
    val idx = formatFieldGIndex(i)
    if (idx >= 0) {
      rvb.setFieldIndex(idx)
      rvb.setMissing()
    }
  }

  def parse(l: VCFLine, rvb: RegionValueBuilder) {
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

class ParseLineContext(val rowType: TStruct, val infoFlagFieldNames: java.util.HashSet[String], val nSamples: Int) {
  val entryType: TStruct = rowType.fieldOption(LowerMatrixIR.entriesFieldName) match {
    case Some(entriesArray) => entriesArray.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
    case None => TStruct.empty
  }
  val infoSignature = rowType.fieldOption("info").map(_.typ.asInstanceOf[TStruct]).orNull
  val hasQual = rowType.hasField("qual")
  val hasFilters = rowType.hasField("filters")
  val hasEntryFields = entryType.size > 0

  val infoFields: java.util.HashMap[String, Int] = if (infoSignature != null) makeJavaMap(infoSignature.fieldIdx) else null
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
  def warnDuplicates(ids: Array[String]) {
    val duplicates = ids.counter().filter(_._2 > 1)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.size } duplicate ${ plural(duplicates.size, "sample ID") }:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
    }
  }

  def globAllVCFs(arguments: Array[String],
    fs: FS,
    forceGZ: Boolean = false,
    gzAsBGZ: Boolean = false): Array[FileStatus] = {
    val statuses = fs.globAllStatuses(arguments)

    if (statuses.isEmpty)
      fatal("arguments refer to no files")

    statuses.foreach { status =>
      val file = status.getPath
      if (!(file.endsWith(".vcf") || file.endsWith(".vcf.bgz") || file.endsWith(".vcf.gz")))
        warn(s"expected input file '$file' to end in .vcf[.bgz, .gz]")
      if (file.endsWith(".gz"))
        checkGzippedFile(fs, file, forceGZ, gzAsBGZ)
    }
    statuses
  }

  def getEntryFloatType(entryFloatTypeName: String): TNumeric = {
    IRParser.parseType(entryFloatTypeName) match {
      case TFloat32 => TFloat32
      case TFloat64 => TFloat64
      case _ => fatal(
        s"""invalid floating point type:
        |  expected ${TFloat32._toPretty} or ${TFloat64._toPretty}, got ${entryFloatTypeName}"""
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
    line: VCFCompoundHeaderLine,
    callFields: Set[String],
    floatType: TNumeric,
    arrayElementsRequired: Boolean = false
  ): ((String, PType), (String, Map[String, String]), Boolean) = {
    val id = line.getID
    val isCall = id == "GT" || callFields.contains(id)

    val baseType = (line.getType, isCall) match {
      case (VCFHeaderLineType.Integer, false) => PInt32()
      case (VCFHeaderLineType.Float, false) => PType.canonical(floatType)
      case (VCFHeaderLineType.String, true) => PCanonicalCall()
      case (VCFHeaderLineType.String, false) => PCanonicalString()
      case (VCFHeaderLineType.Character, false) => PCanonicalString()
      case (VCFHeaderLineType.Flag, false) => PBoolean(true)
      case (_, true) => fatal(s"Can only convert a header line with type 'String' to a call type. Found '${ line.getType }'.")
    }

    val attrs = Map("Description" -> line.getDescription,
      "Number" -> headerNumberToString(line),
      "Type" -> headerTypeToString(line))

    val isFlag = line.getType == VCFHeaderLineType.Flag

    if (line.isFixedCount &&
      (line.getCount == 1 ||
        (isFlag && line.getCount == 0)))
      ((id, baseType), (id, attrs), isFlag)
    else if (baseType.isInstanceOf[PCall])
      fatal("fields in 'call_fields' must have 'Number' equal to 1.")
    else
      ((id, PCanonicalArray(baseType.setRequired(arrayElementsRequired))), (id, attrs), isFlag)
  }

  def headerSignature[T <: VCFCompoundHeaderLine](
    lines: java.util.Collection[T],
    callFields: Set[String],
    floatType: TNumeric,
    arrayElementsRequired: Boolean = false
  ): (PStruct, VCFAttributes, Set[String]) = {
    val (fields, attrs, flags) = lines.asScala
      .map { line => headerField(line, callFields, floatType, arrayElementsRequired) }
      .unzip3

    val flagFieldNames = fields.zip(flags)
      .flatMap { case ((f, _), isFlag) => if (isFlag) Some(f) else None }
      .toSet

    (PCanonicalStruct(true, fields.toArray: _*), attrs.toMap, flagFieldNames)
  }

  def parseHeader(
    callFields: Set[String],
    floatType: TNumeric,
    lines: Array[String],
    arrayElementsRequired: Boolean = true
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
    val (infoSignature, infoAttrs, infoFlagFields) = headerSignature(infoHeader, callFields, TFloat64)

    val formatHeader = header.getFormatHeaderLines
    val (gSignature, formatAttrs, _) = headerSignature(formatHeader, callFields, floatType, arrayElementsRequired = arrayElementsRequired)

    val vaSignature = PCanonicalStruct(Array(
      PField("rsid", PCanonicalString(), 0),
      PField("qual", PFloat64(), 1),
      PField("filters", PCanonicalSet(PCanonicalString(true)), 2),
      PField("info", infoSignature, 3)), true)

    val headerLine = lines.last
    if (!(headerLine(0) == '#' && headerLine(1) != '#'))
      fatal(
        s"""corrupt VCF: expected final header line of format '#CHROM\tPOS\tID...'
           |  found: @1""".stripMargin, headerLine)

    val sampleIds: Array[String] = headerLine.split("\t").drop(9)

    VCFHeaderInfo(
      sampleIds,
      infoSignature,
      vaSignature,
      gSignature,
      filterAttrs,
      infoAttrs,
      formatAttrs,
      infoFlagFields)
  }

  def getHeaderLines[T](
    fs: FS,
    file: String,
    filterAndReplace: TextInputFilterAndReplace): Array[String] = fs.readLines(file, filterAndReplace) { lines =>
      lines
      .takeWhile { line => line.value(0) == '#' }
      .map(_.value)
      .toArray
  }

  def parseLine(
    rgBc: Option[BroadcastValue[ReferenceGenome]],
    contigRecoding: Map[String, String],
    skipInvalidLoci: Boolean,
    rowPType: PStruct,
    rvb: RegionValueBuilder,
    parseLineContext: ParseLineContext,
    vcfLine: VCFLine): Boolean = {
    val hasLocus = rowPType.hasField("locus")
    val hasAlleles = rowPType.hasField("alleles")
    val hasRSID = rowPType.hasField("rsid")
    val hasEntries = rowPType.hasField(LowerMatrixIR.entriesFieldName)

    rvb.start(rowPType)
    rvb.startStruct()
    val present = vcfLine.parseAddVariant(rvb, rgBc.map(_.value), contigRecoding, hasLocus, hasAlleles, hasRSID, skipInvalidLoci)
    if (!present)
      return present

    parseLine(parseLineContext, vcfLine, rvb, !hasEntries)
    true
  }

  // parses the Variant (key), and ID if necessary, leaves the rest to f
  def parseLines[C](
    makeContext: () => C
  )(f: (C, VCFLine, RegionValueBuilder) => Unit
  )(lines: ContextRDD[WithContext[String]],
    rowPType: PStruct,
    rgBc: Option[BroadcastValue[ReferenceGenome]],
    contigRecoding: Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean
  ): ContextRDD[Long] = {
    val hasRSID = rowPType.hasField("rsid")
    lines.cmapPartitions { (ctx, it) =>
      new Iterator[Long] {
        val rvb = ctx.rvb
        var ptr: Long = 0

        val context: C = makeContext()

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
              val vcfLine = new VCFLine(line, arrayElementsRequired, abs, abi, abf, abd)
              rvb.start(rowPType)
              rvb.startStruct()
              present = vcfLine.parseAddVariant(rvb, rgBc.map(_.value), contigRecoding, hasRSID, true, true, skipInvalidLoci)
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
                  .map { c => if (c == '\t') ' ' else c }

                val prefix = if (excerptStart > 0) "... " else ""
                val suffix = if (excerptEnd < line.length) " ..." else ""

                var caretPad = prefix.length + pos - excerptStart
                var pad = " " * caretPad

                fatal(s"${ source.locationString(pos) }: ${ e.msg }\n$prefix$excerpt$suffix\n$pad^\noffending line: @1\nsee the Hail log for the full offending line", line, e)
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

  def parseHeaderMetadata(fs: FS, callFields: Set[String], entryFloatType: TNumeric, headerFile: String): VCFMetadata = {
    val headerLines = getHeaderLines(fs, headerFile, TextInputFilterAndReplace())
    val VCFHeaderInfo(_, _, _, _, filterAttrs, infoAttrs, formatAttrs, _) = parseHeader(callFields, entryFloatType, headerLines)

    Map("filter" -> filterAttrs, "info" -> infoAttrs, "format" -> formatAttrs)
  }

  def parseLine(
    c: ParseLineContext,
    l: VCFLine,
    rvb: RegionValueBuilder,
    dropSamples: Boolean = false
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

case class PartitionedVCFPartition(index: Int, chrom: String, start: Int, end: Int) extends Partition

class PartitionedVCFRDD(
  fsBc: BroadcastValue[FS],
  file: String,
  @(transient@param) reverseContigMapping: Map[String, String],
  @(transient@param) _partitions: Array[Partition]
) extends RDD[String](SparkBackend.sparkContext("PartitionedVCFRDD"), Seq()) {

  val contigRemappingBc = if (reverseContigMapping.size != 0) sparkContext.broadcast(reverseContigMapping) else null

  protected def getPartitions: Array[Partition] = _partitions

  def compute(split: Partition, context: TaskContext): Iterator[String] = {
    val p = split.asInstanceOf[PartitionedVCFPartition]

    val chromToQuery = if (contigRemappingBc != null) contigRemappingBc.value.getOrElse(p.chrom, p.chrom) else p.chrom

    val reg = {
      val r = new TabixReader(file, fsBc.value)
      val tid = r.chr2tid(chromToQuery)
      r.queryPairs(tid, p.start - 1, p.end)
    }
    if (reg.isEmpty)
      return Iterator.empty

    val lines = new TabixLineIterator(fsBc, file, reg)

    // clean up
    val context = TaskContext.get
    context.addTaskCompletionListener[Unit] { (context: TaskContext) =>
      lines.close()
    }

    val it: Iterator[String] = new Iterator[String] {
      private var l = lines.next()

      def hasNext: Boolean = l != null

      def next(): String = {
        assert(l != null)
        val n = l
        l = lines.next()
        if (l == null)
          lines.close()
        n
      }
    }

    it.filter { l =>
      val t1 = l.indexOf('\t')
      val t2 = l.indexOf('\t', t1 + 1)

      val chrom = l.substring(0, t1)
      val pos = l.substring(t1 + 1, t2).toInt

      if (chrom != chromToQuery) {
        throw new RuntimeException(s"bad chromosome! ${chromToQuery}, $l")
      }
      p.start <= pos && pos <= p.end
    }
  }
}

object MatrixVCFReader {
  def apply(ctx: ExecuteContext,
    files: Seq[String],
    callFields: Set[String],
    entryFloatTypeName: String,
    headerFile: Option[String],
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
    partitionsJSON: String): MatrixVCFReader = {
    MatrixVCFReader(ctx, MatrixVCFReaderParameters(
      files, callFields, entryFloatTypeName, headerFile, nPartitions, blockSizeInMB, minPartitions, rg,
      contigRecoding, arrayElementsRequired, skipInvalidLoci, gzAsBGZ, forceGZ, filterAndReplace,
      partitionsJSON))
  }

  def apply(ctx: ExecuteContext, params: MatrixVCFReaderParameters): MatrixVCFReader = {
    val backend = ctx.backend
    val fs = ctx.fs
    val fsBc = fs.broadcast

    val referenceGenome = params.rg.map(ReferenceGenome.getReference)

    referenceGenome.foreach(_.validateContigRemap(params.contigRecoding))

    val fileStatuses = LoadVCF.globAllVCFs(fs.globAll(params.files), fs, params.forceGZ, params.gzAsBGZ)

    val entryFloatType = LoadVCF.getEntryFloatType(params.entryFloatTypeName)

    val headerLines1 = getHeaderLines(fs, params.headerFile.getOrElse(fileStatuses.head.getPath), params.filterAndReplace)
    val header1 = parseHeader(params.callFields, entryFloatType, headerLines1, arrayElementsRequired = params.arrayElementsRequired)

    if (fileStatuses.length > 1) {
      if (params.headerFile.isEmpty) {
        if (backend.isInstanceOf[SparkBackend]) {
          val header1Bc = backend.broadcast(header1)

          val localCallFields = params.callFields
          val localFloatType = entryFloatType
          val files = fileStatuses.map(_.getPath)
          val localArrayElementsRequired = params.arrayElementsRequired
          val localFilterAndReplace = params.filterAndReplace
          SparkBackend.sparkContext("MatrixVCFReader.apply").parallelize(files.tail, math.max(1, files.length - 1)).foreach { file =>
            val fs = fsBc.value
            val hd = parseHeader(
              localCallFields, localFloatType, getHeaderLines(fs, file, localFilterAndReplace),
              arrayElementsRequired = localArrayElementsRequired)
            val hd1 = header1Bc.value

            if (hd1.sampleIds.length != hd.sampleIds.length) {
              fatal(
                s"""invalid sample IDs: expected same number of samples for all inputs.
                   | ${ files(0) } has ${ hd1.sampleIds.length } ids and
                   | ${ file } has ${ hd.sampleIds.length } ids.
           """.stripMargin)
            }

            hd1.sampleIds.iterator.zipAll(hd.sampleIds.iterator, None, None)
              .zipWithIndex.foreach { case ((s1, s2), i) =>
              if (s1 != s2) {
                fatal(
                  s"""invalid sample IDs: expected sample ids to be identical for all inputs. Found different sample IDs at position $i.
                     |    ${ files(0) }: $s1
                     |    $file: $s2""".stripMargin)
              }
            }

            if (hd1.genotypeSignature != hd.genotypeSignature)
              fatal(
                s"""invalid genotype signature: expected signatures to be identical for all inputs.
                   |   ${ files(0) }: ${ hd1.genotypeSignature.toString }
                   |   $file: ${ hd.genotypeSignature.toString }""".stripMargin)

            if (hd1.vaSignature != hd.vaSignature)
              fatal(
                s"""invalid variant annotation signature: expected signatures to be identical for all inputs.
                   |   ${ files(0) }: ${ hd1.vaSignature.toString }
                   |   $file: ${ hd.vaSignature.toString }""".stripMargin)
          }
        } else {
          warn("Non-Spark backend: not verifying agreement of headers between input VCF files.")
        }
      } else {
        warn("Loading user-provided header file. The sample IDs, " +
          "INFO fields, and FORMAT fields were not checked for agreement with input data.")
      }
    }

    val VCFHeaderInfo(sampleIDs, infoSignature, vaSignature, genotypeSignature, _, _, _, infoFlagFieldNames) = header1

    LoadVCF.warnDuplicates(sampleIDs)

    val locusType = TLocus.schemaFromRG(referenceGenome)
    val kType = TStruct("locus" -> locusType, "alleles" -> TArray(TString))

    val fullMatrixType: MatrixType = MatrixType(
      TStruct.empty,
      colType = TStruct("s" -> TString),
      colKey = Array("s"),
      rowType = kType ++ vaSignature.virtualType,
      rowKey = Array("locus", "alleles"),
      // rowKey = Array.empty[String],
      entryType = genotypeSignature.virtualType)

    val fullType: TableType = fullMatrixType.canonicalTableType

    val fullRVDType = RVDType(PCanonicalStruct(true,
      FastIndexedSeq(("locus", PCanonicalLocus.schemaFromRG(referenceGenome, true)),
        ("alleles", PCanonicalArray(PCanonicalString(true), true))) ++
        header1.vaSignature.fields.map { f => (f.name, f.typ) } ++
        Array(LowerMatrixIR.entriesFieldName -> PCanonicalArray(header1.genotypeSignature, true)): _*),
      fullType.key)

    new MatrixVCFReader(params, fileStatuses, infoFlagFieldNames, referenceGenome, fullMatrixType, fullRVDType, sampleIDs)
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
  partitionsJSON: String)

class MatrixVCFReader(
  val params: MatrixVCFReaderParameters,
  fileStatuses: IndexedSeq[FileStatus],
  infoFlagFieldNames: Set[String],
  referenceGenome: Option[ReferenceGenome],
  val fullMatrixType: MatrixType,
  fullRVDType: RVDType,
  sampleIDs: Array[String]
) extends MatrixHybridReader {

  def pathsUsed: Seq[String] = params.files

  def nCols: Int = sampleIDs.length

  val columnCount: Option[Int] = Some(nCols)

  val partitionCounts: Option[IndexedSeq[Long]] = None

  def rowAndGlobalPTypes(context: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    coerce[PStruct](fullRVDType.rowType.subsetTo(requestedType.rowType)) ->
      PType.canonical(requestedType.globalType).asInstanceOf[PStruct]
  }

  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val fs = ctx.fs

    val rgBc = referenceGenome.map(_.broadcast)
    val localArrayElementsRequired = params.arrayElementsRequired
    val localContigRecording = params.contigRecoding
    val localSkipInvalidLoci = params.skipInvalidLoci
    val localInfoFlagFieldNames = infoFlagFieldNames
    val localNSamples = nCols
    val localFilterAndReplace = params.filterAndReplace

    val tt = fullMatrixType.toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)

    val lines = GenericLines.read(fs, fileStatuses, params.nPartitions, params.blockSizeInMB, params.minPartitions, params.gzAsBGZ, params.forceGZ)

    val globals = Row(sampleIDs.map(Row(_)).toFastIndexedSeq)

    val fullRowPType: PType = fullRVDType.rowType

    val bodyPType = (requestedRowType: TStruct) => fullRowPType.subsetTo(requestedRowType).asInstanceOf[PStruct]

    val linesBody = lines.body
    val body = { (requestedType: TStruct) =>
      val requestedPType = bodyPType(requestedType)

      { (region: Region, context: Any) =>
        val parseLineContext = new ParseLineContext(requestedType, makeJavaSet(localInfoFlagFieldNames), localNSamples)

        val rvb = new RegionValueBuilder(region)

        val abs = new MissingArrayBuilder[String]
        val abi = new MissingArrayBuilder[Int]
        val abf = new MissingArrayBuilder[Float]
        val abd = new MissingArrayBuilder[Double]

        val transformer = localFilterAndReplace.transformer()

        linesBody(context)
          .filter { line =>
            val text = line.toString
            val newText = transformer(text)
            if (newText != null) {
              rvb.clear()
              try {
                val vcfLine = new VCFLine(newText, localArrayElementsRequired, abs, abi, abf, abd)
                LoadVCF.parseLine(rgBc, localContigRecording, localSkipInvalidLoci,
                  requestedPType, rvb, parseLineContext, vcfLine)
              } catch {
                case e: Exception =>
                  fatal(s"${ line.file }:offset ${ line.offset }: error while parsing line\n" +
                    s"$newText\n", e)
              }
            } else
              false
          }.map { _ =>
          rvb.result().offset
        }
      }
    }

    new GenericTableValue(
      tt,
      None,
      { (requestedGlobalsType: Type) =>
        val subset = tt.globalType.valueSubsetter(requestedGlobalsType)
        subset(globals).asInstanceOf[Row]
      },
      lines.contextType,
      lines.contexts,
      bodyPType,
      body)
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType)

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue =
    executeGeneric(ctx).toTableValue(ctx ,tr.typ)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixVCFReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixVCFReader => params == that.params
    case _ => false
  }
}

class VCFsReader(
  ctx: ExecuteContext,
  files: Array[String],
  callFields: Set[String],
  entryFloatTypeName: String,
  rg: Option[String],
  contigRecoding: Map[String, String],
  arrayElementsRequired: Boolean,
  skipInvalidLoci: Boolean,
  gzAsBGZ: Boolean,
  forceGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  partitionsJSON: String, partitionsTypeStr: String,
  externalSampleIds: Option[Array[Array[String]]],
  externalHeader: Option[String]) {

  require(!(externalSampleIds.isEmpty ^ externalHeader.isEmpty))

  private val backend = ctx.backend
  private val fs = ctx.fs
  private val fsBc = fs.broadcast

  private val referenceGenome = rg.map(ReferenceGenome.getReference)

  referenceGenome.foreach(_.validateContigRemap(contigRecoding))

  val reverseContigMapping: Map[String, String] =
    contigRecoding.toArray
      .groupBy(_._2)
      .map { case (target, mappings) =>
        if (mappings.length > 1)
          fatal(s"contig_recoding may not map multiple contigs to the same target contig, " +
            s"due to ambiguity when querying the tabix index." +
            s"\n  Duplicate mappings: ${ mappings.map(_._1).mkString(",") } all map to ${ target }")
        (target, mappings.head._1)
      }.toMap

  private val locusType = TLocus.schemaFromRG(referenceGenome)
  private val rowKeyType = TStruct("locus" -> locusType)

  private val file1 = files.head
  private val headerLines1 = getHeaderLines(fs, externalHeader.getOrElse(file1), filterAndReplace)
  private val headerLines1Bc = backend.broadcast(headerLines1)
  private val entryFloatType = LoadVCF.getEntryFloatType(entryFloatTypeName)
  private val header1 = parseHeader(callFields, entryFloatType, headerLines1, arrayElementsRequired = arrayElementsRequired)

  private val kType = TStruct("locus" -> locusType, "alleles" -> TArray(TString))

  val typ = MatrixType(
    TStruct.empty,
    colType = TStruct("s" -> TString),
    colKey = Array("s"),
    rowType = kType ++ header1.vaSignature.virtualType,
    rowKey = Array("locus"),
    entryType = header1.genotypeSignature.virtualType)

  val fullRVDType = RVDType(PCanonicalStruct(true,
    FastIndexedSeq(("locus", PCanonicalLocus.schemaFromRG(referenceGenome, true)), ("alleles", PCanonicalArray(PCanonicalString(true), true)))
      ++ header1.vaSignature.fields.map { f => (f.name, f.typ) }
      ++ Array(LowerMatrixIR.entriesFieldName -> PCanonicalArray(header1.genotypeSignature, true)): _*),
    typ.rowKey)

  val partitioner: RVDPartitioner = {
    val partitionsType = IRParser.parseType(partitionsTypeStr)
    val jv = JsonMethods.parse(partitionsJSON)
    val rangeBounds = JSONAnnotationImpex.importAnnotation(jv, partitionsType)
      .asInstanceOf[IndexedSeq[Interval]]

    rangeBounds.zipWithIndex.foreach { case (b, i) =>
      if (!(b.includesStart && b.includesEnd))
        fatal("range bounds must be inclusive")

      val start = b.start.asInstanceOf[Row].getAs[Locus](0)
      val end = b.end.asInstanceOf[Row].getAs[Locus](0)
      if (start.contig != end.contig)
        fatal(s"partition spec must not cross contig boundaries, start: ${start.contig} | end: ${end.contig}")
    }

    new RVDPartitioner(
      Array("locus"),
      rowKeyType,
      rangeBounds)
  }

  val partitions = partitioner.rangeBounds.zipWithIndex.map { case (b, i) =>
    val start = b.start.asInstanceOf[Row].getAs[Locus](0)
    val end = b.end.asInstanceOf[Row].getAs[Locus](0)
    PartitionedVCFPartition(i, start.contig, start.position, end.position): Partition
  }

  private val fileInfo: Array[Array[String]] = externalSampleIds.getOrElse {
    val localBcFS = fsBc
    val localFile1 = file1
    val localEntryFloatType = entryFloatType
    val localCallFields = callFields
    val localArrayElementsRequired = arrayElementsRequired
    val localFilterAndReplace = filterAndReplace
    val localGenotypeSignature = header1.genotypeSignature
    val localVASignature = header1.vaSignature

    SparkBackend.sparkContext("VCFsReader.fileInfo").parallelize(files, files.length).map { file =>
      val fs = localBcFS.value
      val headerLines = getHeaderLines(fs, file, localFilterAndReplace)
      val header = parseHeader(
        localCallFields, localEntryFloatType, headerLines, arrayElementsRequired = localArrayElementsRequired)

      if (header.genotypeSignature != localGenotypeSignature)
        fatal(
          s"""invalid genotype signature: expected signatures to be identical for all inputs.
             |   $localFile1: $localGenotypeSignature
             |   $file: ${header.genotypeSignature}""".stripMargin)

      if (header.vaSignature != localVASignature)
        fatal(
          s"""invalid variant annotation signature: expected signatures to be identical for all inputs.
             |   $localFile1: $localVASignature
             |   $file: ${header.vaSignature}""".stripMargin)


      header.sampleIds
    }
      .collect()
  }

  def readFile(ctx: ExecuteContext, file: String, i: Int): MatrixIR = {
    val sampleIDs = fileInfo(i)
    val localInfoFlagFieldNames = header1.infoFlagFields
    val localTyp = typ
    val tt = localTyp.canonicalTableType
    val rvdType = fullRVDType

    val lines = ContextRDD.weaken(
      new PartitionedVCFRDD(ctx.fsBc, file, reverseContigMapping, partitions)
        .map(l =>
          WithContext(l, Context(l, file, None))))

    val parsedLines = parseLines { () =>
      new ParseLineContext(tt.rowType,
        makeJavaSet(localInfoFlagFieldNames),
        sampleIDs.length)
    } { (c, l, rvb) =>
      LoadVCF.parseLine(c, l, rvb)
    }(lines, rvdType.rowType, referenceGenome.map(_.broadcast), contigRecoding, arrayElementsRequired, skipInvalidLoci)

    val rvd = RVD(rvdType,
      partitioner,
      parsedLines)

    MatrixLiteral(ctx, typ, rvd, Row.empty, sampleIDs.map(Row(_)))
  }

  def read(ctx: ExecuteContext): Array[MatrixIR] = {
    files.zipWithIndex.map { case (file, i) =>
      readFile(ctx, file, i)
    }
  }
}

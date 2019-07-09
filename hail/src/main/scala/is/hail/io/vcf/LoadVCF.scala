package is.hail.io.vcf

import htsjdk.variant.vcf._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.BroadcastValue
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.{ExecuteContext, IRParser, LowerMatrixIR, MatrixHybridReader, MatrixIR, MatrixLiteral, MatrixValue, PruneDeadFields, TableRead, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.io.tabix._
import is.hail.io.vcf.LoadVCF.{getHeaderLines, parseHeader, parseLines}
import is.hail.io.{VCFAttributes, VCFMetadata}
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.json4s.JsonAST.{JInt, JObject, JString}
import org.json4s.jackson.JsonMethods

import scala.annotation.meta.param
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.implicitConversions

import is.hail.io.fs.FS

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() {
    throw new UnsupportedOperationException
  }
}

case class VCFHeaderInfo(sampleIds: Array[String], infoSignature: TStruct, vaSignature: TStruct, genotypeSignature: TStruct,
  filtersAttrs: VCFAttributes, infoAttrs: VCFAttributes, formatAttrs: VCFAttributes, infoFlagFields: Set[String])

class VCFParseError(val msg: String, val pos: Int) extends RuntimeException(msg)

final class VCFLine(val line: String, arrayElementsRequired: Boolean) {
  var pos: Int = 0

  val abs = new MissingArrayBuilder[String]
  val abi = new MissingArrayBuilder[Int]
  val abf = new MissingArrayBuilder[Float]
  val abd = new MissingArrayBuilder[Double]

  def parseError(msg: String): Unit = throw new VCFParseError(msg, pos)

  def numericValue(c: Char): Int = {
    if (c < '0' || c > '9')
      parseError(s"invalid character '$c' in integer literal")
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
    assert(line(pos) == '\t')
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

  // return false if it should be filtered
  def parseAddVariant(
    rvb: RegionValueBuilder,
    rg: Option[ReferenceGenome],
    contigRecoding: Map[String, String],
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
    if (!htsjdk.variant.variantcontext.Allele.acceptableAlleleBases(ref, true))
      return false
    nextField()

    // ALT
    parseAltAlleles()
    nextField()

    rvb.startStruct() // pk: Locus
    rvb.addString(recodedContig)
    rvb.addInt(start)
    rvb.endStruct()

    rvb.startArray(abs.length + 1) // ref plus alts
    rvb.addString(ref)
    var i = 0
    while (i < abs.length) {
      rvb.addString(abs(i))
      i += 1
    }
    rvb.endArray()

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
        rvb.addInt(Call1(j, phased = false))
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
      rvb.addInt(Call2(j, k, phased = isPhased))
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
        case TInt32(_) => parseAddInfoInt(rvb)
        case TString(_) => parseAddInfoString(rvb)
        case TFloat64(_) => parseAddInfoDouble(rvb)
        case TArray(TInt32(_), _) => parseAddInfoArrayInt(rvb)
        case TArray(TFloat64(_), _) => parseAddInfoArrayDouble(rvb)
        case TArray(TString(_), _) => parseAddInfoArrayString(rvb)
      }
    }
  }

  def parseAddInfo(rvb: RegionValueBuilder, infoType: TStruct, infoFlagFieldNames: Set[String]) {
    def addField(key: String) = {
      if (infoType.hasField(key)) {
        rvb.setFieldIndex(infoType.fieldIdx(key))
        if (infoFlagFieldNames.contains(key)) {
          if (line(pos) == '=') {
            pos += 1
            val s = parseInfoString()
            if (s != "0")
              rvb.addBoolean(true)
          } else
            rvb.addBoolean(true)
        } else
          parseAddInfoField(rvb, infoType.fieldType(key))
      }
    }
    rvb.startStruct()
    infoType.fields.foreach { f =>
      if (infoFlagFieldNames.contains(f.name))
        rvb.addBoolean(false)
      else
        rvb.setMissing()
    }

    // handle first key, which may be '.' for missing info
    var key = parseInfoKey()
    if (key == ".") {
      if (endField()) {
        rvb.endStruct()
        return
      } else
        parseError(s"invalid INFO key $key")
    }

    addField(key)
    skipInfoField()

    while (!endField()) {
      nextInfoField()
      key = parseInfoKey()
      if (key == ".") {
        parseError(s"invalid INFO key $key")
      }
      addField(key)
      skipInfoField()
    }

    rvb.setFieldIndex(infoType.size)
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

class FormatParser(
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
        case TCall(_) =>
          l.parseAddCall(rvb)
        case TInt32(_) =>
          l.parseAddFormatInt(rvb)
        case TFloat32(_) =>
          l.parseAddFormatFloat(rvb)
        case TFloat64(_) =>
          l.parseAddFormatDouble(rvb)
        case TString(_) =>
          l.parseAddFormatString(rvb)
        case TArray(TInt32(_), _) =>
          l.parseAddFormatArrayInt(rvb)
        case TArray(TFloat32(_), _) =>
          l.parseAddFormatArrayFloat(rvb)
        case TArray(TFloat64(_), _) =>
          l.parseAddFormatArrayDouble(rvb)
        case TArray(TString(_), _) =>
          l.parseAddFormatArrayString(rvb)
      }
    }
  }

  def setFieldMissing(rvb: RegionValueBuilder, i: Int) {
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
        setFieldMissing(rvb, i)
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

class ParseLineContext(typ: TableType, val infoFlagFieldNames: Set[String], val nSamples: Int) {
  val entryType: TStruct = typ.rowType.fieldOption(LowerMatrixIR.entriesFieldName) match {
    case Some(entriesArray) => entriesArray.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
    case None => TStruct()
  }
  val infoSignature = typ.rowType.fieldOption("info").map(_.typ.asInstanceOf[TStruct]).orNull
  val hasQual = typ.rowType.hasField("qual")
  val hasFilters = typ.rowType.hasField("filters")
  val hasEntryFields = entryType.size > 0

  val formatParsers = mutable.Map[String, FormatParser]()

  def getFormatParser(format: String): FormatParser = {
    formatParsers.get(format) match {
      case Some(fp) => fp
      case None =>
        val fp = FormatParser(entryType, format)
        formatParsers += format -> fp
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
    gzAsBGZ: Boolean = false): Array[String] = {
    val inputs = fs.globAll(arguments)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!(input.endsWith(".vcf") || input.endsWith(".vcf.bgz") || input.endsWith(".vcf.gz")))
        warn(s"expected input file '$input' to end in .vcf[.bgz, .gz]")
      if (input.endsWith(".gz"))
        checkGzippedFile(fs, input, forceGZ, gzAsBGZ)
    }
    inputs
  }

  def getEntryFloatType(entryFloatTypeName: String): TNumeric = {
    IRParser.parseType(entryFloatTypeName) match {
      case t32: TFloat32 => t32
      case t64: TFloat64 => t64
      case _ => fatal(
        s"""invalid floating point type:
        |  expected ${TFloat32()._toPretty} or ${TFloat64()._toPretty}, got ${entryFloatTypeName}"""
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
    i: Int,
    callFields: Set[String],
    floatType: TNumeric,
    arrayElementsRequired: Boolean = false
  ): (Field, (String, Map[String, String]), Boolean) = {
    val id = line.getID
    val isCall = id == "GT" || callFields.contains(id)

    val baseType = (line.getType, isCall) match {
      case (VCFHeaderLineType.Integer, false) => TInt32()
      case (VCFHeaderLineType.Float, false) => floatType
      case (VCFHeaderLineType.String, true) => TCall()
      case (VCFHeaderLineType.String, false) => TString()
      case (VCFHeaderLineType.Character, false) => TString()
      case (VCFHeaderLineType.Flag, false) => TBoolean()
      case (_, true) => fatal(s"Can only convert a header line with type 'String' to a call type. Found '${ line.getType }'.")
    }

    val attrs = Map("Description" -> line.getDescription,
      "Number" -> headerNumberToString(line),
      "Type" -> headerTypeToString(line))

    val isFlag = line.getType == VCFHeaderLineType.Flag

    if (line.isFixedCount &&
      (line.getCount == 1 ||
        (isFlag && line.getCount == 0)))
      (Field(id, baseType, i), (id, attrs), isFlag)
    else if (baseType.isInstanceOf[TCall])
      fatal("fields in 'call_fields' must have 'Number' equal to 1.")
    else
      (Field(id, TArray(baseType.setRequired(arrayElementsRequired)), i), (id, attrs), isFlag)
  }

  def headerSignature[T <: VCFCompoundHeaderLine](
    lines: java.util.Collection[T],
    callFields: Set[String],
    floatType: TNumeric,
    arrayElementsRequired: Boolean = false
  ): (TStruct, VCFAttributes, Set[String]) = {
    val (fields, attrs, flags) = lines
      .zipWithIndex
      .map { case (line, i) => headerField(line, i, callFields, floatType, arrayElementsRequired) }
      .unzip3

    val flagFieldNames = fields.zip(flags)
      .flatMap { case (f, isFlag) => if (isFlag) Some(f.name) else None }
      .toSet

    (TStruct(fields.toArray), attrs.toMap, flagFieldNames)
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

    // FIXME apply descriptions when HTSJDK is fixed to expose filter descriptions
    val filterAttrs: VCFAttributes = header
      .getFilterLines
      .toList
      // (ID, description)
      .map(line => (line.getID, Map("Description" -> "")))
      .toMap

    val infoHeader = header.getInfoHeaderLines
    val (infoSignature, infoAttrs, infoFlagFields) = headerSignature(infoHeader, callFields, TFloat64())

    val formatHeader = header.getFormatHeaderLines
    val (gSignature, formatAttrs, _) = headerSignature(formatHeader, callFields, floatType, arrayElementsRequired = arrayElementsRequired)

    val vaSignature = TStruct(Array(
      Field("rsid", TString(), 0),
      Field("qual", TFloat64(), 1),
      Field("filters", TSet(TString()), 2),
      Field("info", infoSignature, 3)))

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

  // parses the Variant (key), and ID if necessary, leaves the rest to f
  def parseLines[C](
    makeContext: () => C
  )(f: (C, VCFLine, RegionValueBuilder) => Unit
  )(lines: ContextRDD[RVDContext, WithContext[String]],
    t: Type,
    rgBc: Option[BroadcastValue[ReferenceGenome]],
    contigRecoding: Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean
  ): ContextRDD[RVDContext, RegionValue] = {
    val hasRSID = t.isInstanceOf[TStruct] && t.asInstanceOf[TStruct].hasField("rsid")
    lines.cmapPartitions { (ctx, it) =>
      new Iterator[RegionValue] {
        val region = ctx.region
        val rvb = ctx.rvb
        val rv = RegionValue(region)

        val context: C = makeContext()

        var present: Boolean = false

        def hasNext: Boolean = {
          while (!present && it.hasNext) {
            val lwc = it.next()
            val line = lwc.value
            try {
              val vcfLine = new VCFLine(line, arrayElementsRequired)
              rvb.start(t.physicalType)
              rvb.startStruct()
              present = vcfLine.parseAddVariant(rvb, rgBc.map(_.value), contigRecoding, hasRSID, skipInvalidLoci)
              if (present) {
                f(context, vcfLine, rvb)

                rvb.endStruct()
                rv.setOffset(rvb.end())
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

        def next(): RegionValue = {
          // call hasNext to advance if necessary
          if (!hasNext)
            throw new java.util.NoSuchElementException()
          present = false
          rv
        }
      }
    }
  }

  def parseHeaderMetadata(hc: HailContext, callFields: Set[String], entryFloatType: TNumeric, headerFile: String): VCFMetadata = {
    val fs = hc.sFS
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
      l.parseAddInfo(rvb, c.infoSignature, c.infoFlagFieldNames)
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
  @(transient@param) sc: SparkContext,
  file: String,
  @(transient@param) _partitions: Array[Partition]
) extends RDD[String](sc, Seq()) {
  protected def getPartitions: Array[Partition] = _partitions
  val bcFS = HailContext.bcFS

  def compute(split: Partition, context: TaskContext): Iterator[String] = {
    val p = split.asInstanceOf[PartitionedVCFPartition]

    val reg = {
      val r = new TabixReader(file, bcFS.value)
      val tid = r.chr2tid(p.chrom)
      r.queryPairs(tid, p.start - 1, p.end)
    }
    val lines = new TabixLineIterator(bcFS, file, reg)

    // clean up
    val context = TaskContext.get
    context.addTaskCompletionListener { (context: TaskContext) =>
      lines.close()
    }

    val it = new Iterator[String] {
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
      var t1 = l.indexOf('\t')
      val t2 = l.indexOf('\t', t1 + 1)

      val chrom = l.substring(0, t1)
      val pos = l.substring(t1 + 1, t2).toInt

      assert(chrom == p.chrom)
      p.start <= pos && pos <= p.end
    }
  }
}

case class MatrixVCFReader(
  files: Seq[String],
  callFields: Set[String],
  entryFloatTypeName: String,
  headerFile: Option[String],
  minPartitions: Option[Int],
  rg: Option[String],
  contigRecoding: Map[String, String],
  arrayElementsRequired: Boolean,
  skipInvalidLoci: Boolean,
  gzAsBGZ: Boolean,
  forceGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  partitionsJSON: String) extends MatrixHybridReader {

  private val hc = HailContext.get
  private val sc = hc.sc
  private val fs = hc.sFS
  private val referenceGenome = rg.map(ReferenceGenome.getReference)

  referenceGenome.foreach(_.validateContigRemap(contigRecoding))

  private val inputs = LoadVCF.globAllVCFs(fs.globAll(files), fs, forceGZ, gzAsBGZ)

  private val entryFloatType = LoadVCF.getEntryFloatType(entryFloatTypeName)

  private val headerLines1 = getHeaderLines(fs, headerFile.getOrElse(inputs.head), filterAndReplace)
  private val header1 = parseHeader(callFields, entryFloatType, headerLines1, arrayElementsRequired = arrayElementsRequired)

  if (headerFile.isEmpty) {
    val bcFS = HailContext.bcFS
    val header1Bc = hc.backend.broadcast(header1)

    val localCallFields = callFields
    val localFloatType = entryFloatType
    val localInputs = inputs
    val localArrayElementsRequired = arrayElementsRequired
    val localFilterAndReplace = filterAndReplace
    sc.parallelize(inputs.tail, math.max(1, inputs.length - 1)).foreach { file =>
      val fs = bcFS.value
      val hd = parseHeader(
        localCallFields, localFloatType, getHeaderLines(fs, file, localFilterAndReplace),
        arrayElementsRequired = localArrayElementsRequired)
      val hd1 = header1Bc.value

      if (hd1.sampleIds.length != hd.sampleIds.length) {
        fatal(
          s"""invalid sample IDs: expected same number of samples for all inputs.
             | ${ localInputs(0) } has ${ hd1.sampleIds.length } ids and
             | ${ file } has ${ hd.sampleIds.length } ids.
           """.stripMargin)
      }

      hd1.sampleIds.iterator.zipAll(hd.sampleIds.iterator, None, None)
        .zipWithIndex.foreach { case ((s1, s2), i) =>
        if (s1 != s2) {
          fatal(
            s"""invalid sample IDs: expected sample ids to be identical for all inputs. Found different sample IDs at position $i.
               |    ${ localInputs(0) }: $s1
               |    $file: $s2""".stripMargin)
        }
      }

      if (hd1.genotypeSignature != hd.genotypeSignature)
        fatal(
          s"""invalid genotype signature: expected signatures to be identical for all inputs.
             |   ${ localInputs(0) }: ${ hd1.genotypeSignature.toString }
             |   $file: ${ hd.genotypeSignature.toString }""".stripMargin)

      if (hd1.vaSignature != hd.vaSignature)
        fatal(
          s"""invalid variant annotation signature: expected signatures to be identical for all inputs.
             |   ${ localInputs(0) }: ${ hd1.vaSignature.toString }
             |   $file: ${ hd.vaSignature.toString }""".stripMargin)
    }
  } else {
    warn("Loading user-provided header file. The sample IDs, " +
      "INFO fields, and FORMAT fields were not checked for agreement with input data.")
  }

  private val VCFHeaderInfo(sampleIDs, infoSignature, vaSignature, genotypeSignature, _, _, _, infoFlagFieldNames) = header1

  private val nCols: Int = sampleIDs.length

  val columnCount: Option[Int] = Some(nCols)

  val partitionCounts: Option[IndexedSeq[Long]] = None

  LoadVCF.warnDuplicates(sampleIDs)

  private val locusType = TLocus.schemaFromRG(referenceGenome)
  private val kType = TStruct("locus" -> locusType, "alleles" -> TArray(TString()))

  val fullMatrixType: MatrixType = MatrixType(
    TStruct.empty(),
    colType = TStruct("s" -> TString()),
    colKey = Array("s"),
    rowType = kType ++ vaSignature,
    rowKey = Array("locus", "alleles"),
    entryType = genotypeSignature)

  override lazy val fullType: TableType = fullMatrixType.canonicalTableType

  val fullRVDType: RVDType = RVDType(fullType.rowType.physicalType, fullType.key)

  private lazy val lines = {
    HailContext.maybeGZipAsBGZip(gzAsBGZ) {
      ContextRDD.textFilesLines[RVDContext](sc, inputs, minPartitions, filterAndReplace)
    }
  }

  private lazy val coercer = RVD.makeCoercer(
    fullMatrixType.canonicalTableType.canonicalRVDType,
    1,
    parseLines(
      () => ()
    )((c, l, rvb) => ()
    )(lines,
      fullMatrixType.rowKeyStruct,
      referenceGenome.map(_.broadcast),
      contigRecoding,
      arrayElementsRequired,
      skipInvalidLoci))

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val localCallFields = callFields
    val localFloatType = entryFloatType
    val headerLinesBc = hc.backend.broadcast(headerLines1)

    val requestedType = tr.typ
    assert(PruneDeadFields.isSupertype(requestedType, fullType))

    val localInfoFlagFieldNames = infoFlagFieldNames

    val dropSamples = !requestedType.rowType.hasField(LowerMatrixIR.entriesFieldName)
    val localSampleIDs: Array[String] = if (dropSamples) Array.empty[String] else sampleIDs

    val rvd = if (tr.dropRows)
      RVD.empty(sc, requestedType.canonicalRVDType)
    else
      coercer.coerce(requestedType.canonicalRVDType, parseLines { () =>
        new ParseLineContext(requestedType,
          localInfoFlagFieldNames,
          localSampleIDs.length)
      } { (c, l, rvb) => LoadVCF.parseLine(c, l, rvb, dropSamples) }(
        lines, requestedType.rowType, referenceGenome.map(_.broadcast), contigRecoding, arrayElementsRequired, skipInvalidLoci
      ))

    val globalValue = makeGlobalValue(ctx, requestedType, sampleIDs.map(Row(_)))

    TableValue(requestedType, globalValue, rvd)
  }
}

object ImportVCFs {
  def pyApply(
    files: java.util.List[String],
    callFields: java.util.List[String],
    entryFloatTypeName: String,
    rg: String,
    contigRecoding: java.util.Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean,
    gzAsBGZ: Boolean,
    forceGZ: Boolean,
    partitionsJSON: String,
    partitionsTypeStr: String,
    filter: String,
    find: String,
    replace: String,
    externalSampleIds: java.util.List[java.util.List[String]],
    externalHeader: String
  ): String = {
    val reader = new VCFsReader(
      files.asScala.toArray,
      callFields.asScala.toSet,
      entryFloatTypeName,
      Option(rg),
      Option(contigRecoding).map(_.asScala.toMap).getOrElse(Map.empty[String, String]),
      arrayElementsRequired,
      skipInvalidLoci,
      gzAsBGZ,
      forceGZ,
      TextInputFilterAndReplace(Option(find), Option(filter), Option(replace)),
      partitionsJSON, partitionsTypeStr,
      Option(externalSampleIds).map(_.map(_.asScala.toArray).toArray),
      Option(externalHeader))

    val irArray = reader.read()
    val id = HailContext.get.addIrVector(irArray)
    val sb = new StringBuilder
    val out = JObject(
      "vector_ir_id" -> JInt(id),
      "length" -> JInt(irArray.length),
      "type" -> reader.typ.pyJson)
    JsonMethods.compact(out)
  }
}

class VCFsReader(
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

  private val hc = HailContext.get
  private val backend = HailContext.backend
  private val fs = hc.sFS
  private val bcFS = hc.bcFS
  private val referenceGenome = rg.map(ReferenceGenome.getReference)

  referenceGenome.foreach(_.validateContigRemap(contigRecoding))

  private val locusType = TLocus.schemaFromRG(referenceGenome)
  private val rowKeyType = TStruct("locus" -> locusType)

  private val file1 = files.head
  private val headerLines1 = getHeaderLines(fs, externalHeader.getOrElse(file1), filterAndReplace)
  private val headerLines1Bc = backend.broadcast(headerLines1)
  private val entryFloatType = LoadVCF.getEntryFloatType(entryFloatTypeName)
  private val header1 = parseHeader(callFields, entryFloatType, headerLines1, arrayElementsRequired = arrayElementsRequired)

  private val kType = TStruct("locus" -> locusType, "alleles" -> TArray(TString()))

  val typ = MatrixType(
    TStruct.empty(),
    colType = TStruct("s" -> TString()),
    colKey = Array("s"),
    rowType = kType ++ header1.vaSignature,
    rowKey = Array("locus"),
    entryType = header1.genotypeSignature)

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
    val localBcFS = bcFS
    val localFile1 = file1
    val localEntryFloatType = entryFloatType
    val localCallFields = callFields
    val localArrayElementsRequired = arrayElementsRequired
    val localFilterAndReplace = filterAndReplace
    val localGenotypeSignature = header1.genotypeSignature
    val localVASignature = header1.vaSignature

    hc.sc.parallelize(files, files.length).map { file =>
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


  def readFile(file: String, i: Int): MatrixIR = {
    val sampleIDs = fileInfo(i)
    val localEntryFloatType = entryFloatType
    val localCallFields = callFields
    val localHeaderLines1Bc = headerLines1Bc
    val localInfoFlagFieldNames = header1.infoFlagFields
    val localTyp = typ
    val tt = localTyp.canonicalTableType

    val lines = ContextRDD.weaken[RVDContext](
      new PartitionedVCFRDD(hc.sc, file, partitions)
        .map(l =>
          WithContext(l, Context(l, file, None))))

    val parsedLines = parseLines { () =>
      new ParseLineContext(tt,
        localInfoFlagFieldNames,
        sampleIDs.length)
    } { (c, l, rvb) =>
      LoadVCF.parseLine(c, l, rvb)
    }(lines, tt.rowType, referenceGenome.map(_.broadcast), contigRecoding, arrayElementsRequired, skipInvalidLoci)

    val rvd = RVD(tt.canonicalRVDType,
      partitioner,
      parsedLines)

    MatrixLiteral(typ, rvd, Row.empty, sampleIDs.map(Row(_)))
  }

  def read(): Array[MatrixIR] = {
    files.zipWithIndex.map { case (file, i) =>
      readFile(file, i)
    }
  }
}

package is.hail.io.vcf

import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.vcf._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.{MatrixIR, MatrixLiteral, MatrixRead, MatrixReader, MatrixValue, PruneDeadFields}
import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual._
import is.hail.io._
import is.hail.io.vcf.LoadVCF.{getHeaderLines, parseHeader, parseLines}
import is.hail.io.tabix._
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner, RVDType}
import is.hail.io.{VCFAttributes, VCFMetadata}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

import scala.annotation.meta.param
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
import scala.language.implicitConversions

case class VCFHeaderInfo(sampleIds: Array[String], infoSignature: TStruct, vaSignature: TStruct, genotypeSignature: TStruct,
  filtersAttrs: VCFAttributes, infoAttrs: VCFAttributes, formatAttrs: VCFAttributes, infoFlagFields: Set[String])

class VCFParseError(val msg: String, val pos: Int) extends RuntimeException(msg)

final class VCFLine(val line: String, arrayElementsRequired: Boolean) {
  var pos: Int = 0

  val abs = new MissingArrayBuilder[String]
  val abi = new MissingArrayBuilder[Int]
  val abd = new MissingArrayBuilder[Double]

  def parseError(msg: String): Unit = throw new VCFParseError(msg, pos)

  def numericValue(c: Char): Int = {
    if (c < '0' || c > '9')
      parseError(s"invalid character '$c' in integer literal")
    c - '0'
  }

  // field contexts: field, array field, format field, call field, format array field

  def endField(p: Int): Boolean = {
    p == line.length || line(p) == '\t'
  }

  def endArrayField(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ','
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

  def endFormatArrayField(p: Int): Boolean = {
    if (p == line.length)
      true
    else {
      val c = line(p)
      c == '\t' || c == ':' || c == ','
    }
  }

  def endField(): Boolean = endField(pos)

  def endArrayField(): Boolean = endArrayField(pos)

  def endFormatField(): Boolean = endFormatField(pos)

  def endCallField(): Boolean = endCallField(pos)

  def endFormatArrayField(): Boolean = endFormatArrayField(pos)

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
      endArrayField(pos + 1)
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

  def formatArrayFieldMissing(): Boolean = {
    pos < line.length &&
      line(pos) == '.' &&
      endFormatArrayField(pos + 1)
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
    while (!endArrayField())
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

  def nextField(): Unit = {
    if (pos == line.length)
      parseError("unexpected end of line")
    assert(line(pos) == '\t')
    pos += 1 // tab
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

    skipField() // ID
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
    if (endFormatArrayField())
      parseError("empty integer")
    var mul = 1
    if (line(pos) == '-') {
      mul = -1
      pos += 1
    }
    var v = numericValue(line(pos))
    pos += 1
    while (!endFormatArrayField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v * mul
  }

  def parseStringInFormatArray(): String = {
    val start = pos
    while (!endFormatArrayField())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseDoubleInFormatArray(): Double = {
    val s = parseStringInFormatArray()
    s.toDouble
  }

  def parseArrayElement[T](ab: MissingArrayBuilder[T], eltParser: () => T) {
    if (formatArrayFieldMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      ab.addMissing()
      pos += 1
    } else {
      ab += eltParser()
    }
  }

  def parseIntArrayElement() {
    if (formatArrayFieldMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      abi.addMissing()
      pos += 1
    } else {
      abi += parseIntInFormatArray()
    }
  }

  def parseDoubleArrayElement() {
    if (formatArrayFieldMissing()) {
      if (arrayElementsRequired)
        parseError(s"missing value in FORMAT array. Import with argument 'array_elements_required=False'")
      abd.addMissing()
      pos += 1
    } else {
      abd += parseDoubleInFormatArray()
    }
  }

  def parseStringArrayElement() {
    if (formatArrayFieldMissing()) {
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

class ParseLineContext(typ: MatrixType, val infoFlagFieldNames: Set[String], headerLines: BufferedLineIterator) {
  val gType: TStruct = typ.entryType
  val infoSignature = typ.rowType.fieldOption("info").map(_.typ.asInstanceOf[TStruct]).orNull
  val hasRSID = typ.rowType.hasField("rsid")
  val hasQual = typ.rowType.hasField("qual")
  val hasFilters = typ.rowType.hasField("filters")
  val formatSignature = typ.entryType
  val hasEntryFields = formatSignature.size > 0

  val codec = new htsjdk.variant.vcf.VCFCodec()
  codec.readHeader(headerLines)

  val formatParsers = mutable.Map[String, FormatParser]()

  def getFormatParser(format: String): FormatParser = {
    formatParsers.get(format) match {
      case Some(fp) => fp
      case None =>
        val fp = FormatParser(gType, format)
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
    hConf: hadoop.conf.Configuration,
    forceGZ: Boolean = false,
    gzAsBGZ: Boolean = false): Array[String] = {
    val inputs = hConf.globAll(arguments)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".vcf")
        && !input.endsWith(".vcf.bgz")) {
        if (input.endsWith(".vcf.gz"))
          checkGzippedFile(hConf, input, forceGZ, gzAsBGZ)
        else
          fatal(s"unknown input file type `$input', expect .vcf[.bgz]")
      }
    }
    inputs
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

  def headerField(line: VCFCompoundHeaderLine, i: Int, reader: HtsjdkRecordReader, arrayElementsRequired: Boolean = false): (Field, (String, Map[String, String]), Boolean) = {
    val id = line.getID
    val isCall = id == "GT" || reader.callFields.contains(id)

    val baseType = (line.getType, isCall) match {
      case (VCFHeaderLineType.Integer, false) => TInt32()
      case (VCFHeaderLineType.Float, false) => reader.entryFloatType
      case (VCFHeaderLineType.String, true) => TCall()
      case (VCFHeaderLineType.String, false) => TString()
      case (VCFHeaderLineType.Character, false) => TString()
      case (VCFHeaderLineType.Flag, false) => TBoolean()
      case (_, true) => fatal(s"Can only convert a header line with type `String' to a call type. Found `${ line.getType }'.")
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

  def headerSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    reader: HtsjdkRecordReader, arrayElementsRequired: Boolean = false): (TStruct, VCFAttributes, Set[String]) = {
    val (fields, attrs, flags) = lines
      .zipWithIndex
      .map { case (line, i) => headerField(line, i, reader, arrayElementsRequired) }
      .unzip3

    val flagFieldNames = fields.zip(flags)
      .flatMap { case (f, isFlag) => if (isFlag) Some(f.name) else None }
      .toSet

    (TStruct(fields.toArray), attrs.toMap, flagFieldNames)
  }

  def parseHeader(reader: HtsjdkRecordReader, lines: Array[String], arrayElementsRequired: Boolean = true): VCFHeaderInfo = {
    val codec = new htsjdk.variant.vcf.VCFCodec()
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
    val (infoSignature, infoAttrs, infoFlagFields) = headerSignature(infoHeader, reader)

    val formatHeader = header.getFormatHeaderLines
    val (gSignature, formatAttrs, _) = headerSignature(formatHeader, reader, arrayElementsRequired = arrayElementsRequired)

    val vaSignature = TStruct(Array(
      Field("rsid", TString(), 0),
      Field("qual", TFloat64(), 1),
      Field("filters", TSet(TString()), 2),
      Field("info", infoSignature, 3)))

    val headerLine = lines.last
    if (!(headerLine(0) == '#' && headerLine(1) != '#'))
      fatal(
        s"""corrupt VCF: expected final header line of format `#CHROM\tPOS\tID...'
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
    hConf: Configuration,
    file: String,
    filterAndReplace: TextInputFilterAndReplace): Array[String] = hConf.readLines(file, filterAndReplace) { lines =>
      lines
      .takeWhile { line => line.value(0) == '#' }
      .map(_.value)
      .toArray
  }

  // parses the Variant (key), leaves the rest to f
  def parseLines[C](
    makeContext: () => C
  )(f: (C, VCFLine, RegionValueBuilder) => Unit
  )(lines: ContextRDD[RVDContext, WithContext[String]],
    t: Type,
    rg: Option[ReferenceGenome],
    contigRecoding: Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean
  ): ContextRDD[RVDContext, RegionValue] = {
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
              present = vcfLine.parseAddVariant(rvb, rg, contigRecoding, skipInvalidLoci)
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

  def parseHeaderMetadata(hc: HailContext, reader: HtsjdkRecordReader, headerFile: String): VCFMetadata = {
    val hConf = hc.hadoopConf
    val headerLines = getHeaderLines(hConf, headerFile, TextInputFilterAndReplace())
    val VCFHeaderInfo(_, _, _, _, filterAttrs, infoAttrs, formatAttrs, _) = parseHeader(reader, headerLines)

    Map("filter" -> filterAttrs, "info" -> infoAttrs, "format" -> formatAttrs)
  }

  def parseLine(
    reader: HtsjdkRecordReader,
    c: ParseLineContext,
    l: VCFLine,
    rvb: RegionValueBuilder,
    dropSamples: Boolean = false
  ): Unit = {
    val vc = c.codec.decode(l.line)
    val nSamples = if (!dropSamples) vc.getNSamples else 0

    reader.readVariantInfo(vc, rvb, c.hasRSID, c.hasQual, c.hasFilters, c.infoSignature, c.infoFlagFieldNames)

    rvb.startArray(nSamples) // gs

    if (nSamples > 0) {
      if (!c.hasEntryFields) {
        var i = 0
        while (i < nSamples) {
          rvb.startStruct()
          rvb.endStruct()
          i += 1
        }
      } else {
        // l is pointing at qual
        var i = 0
        while (i < 3) { // qual, filter, info
          l.skipField()
          l.nextField()
          i += 1
        }

        val format = l.parseString()
        l.nextField()

        val fp = c.getFormatParser(format)

        fp.parse(l, rvb)
        i = 1
        while (i < nSamples) {
          l.nextField()
          fp.parse(l, rvb)
          i += 1
        }
      }
    }
    rvb.endArray()
  }
}

case class PartitionedVCFPartition(index: Int, chrom: String, start: Int, end: Int, reg: Array[TbiPair]) extends Partition

class PartitionedVCFRDD(
  @(transient@param) sc: SparkContext,
  file: String,
  @(transient@param) _partitions: Array[Partition]
) extends RDD[String](sc, Seq()) {
  protected def getPartitions: Array[Partition] = _partitions
  val confBc = HailContext.hadoopConfBc

  def compute(split: Partition, context: TaskContext): Iterator[String] = {
    val p = split.asInstanceOf[PartitionedVCFPartition]

    val lines = new TabixLineIterator(confBc, file, p.reg)

    // clean up
    val context = TaskContext.get
    context.addTaskCompletionListener { context =>
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
  entryFloatType: String,
  headerFile: Option[String],
  minPartitions: Option[Int],
  rg: Option[String],
  contigRecoding: Map[String, String],
  arrayElementsRequired: Boolean,
  skipInvalidLoci: Boolean,
  gzAsBGZ: Boolean,
  forceGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  partitionsJSON: String) extends MatrixReader {

  private val hc = HailContext.get
  private val sc = hc.sc
  private val hConf = sc.hadoopConfiguration
  private val referenceGenome = rg.map(ReferenceGenome.getReference)

  referenceGenome.foreach(_.validateContigRemap(contigRecoding))

  private val inputs = LoadVCF.globAllVCFs(hConf.globAll(files), hConf, forceGZ, gzAsBGZ)

  private val reader = new HtsjdkRecordReader(callFields, entryFloatType)

  private val headerLines1 = getHeaderLines(hConf, headerFile.getOrElse(inputs.head), filterAndReplace)
  private val header1 = parseHeader(reader, headerLines1, arrayElementsRequired = arrayElementsRequired)

  if (headerFile.isEmpty) {
    val confBc = HailContext.hadoopConfBc
    val header1Bc = sc.broadcast(header1)

    val localReader = reader
    val localInputs = inputs
    val localArrayElementsRequired = arrayElementsRequired
    val localFilterAndReplace = filterAndReplace
    sc.parallelize(inputs.tail, math.max(1, inputs.length - 1)).foreach { file =>
      val hConf = confBc.value.value
      val hd = parseHeader(localReader, getHeaderLines(hConf, file, localFilterAndReplace), arrayElementsRequired = localArrayElementsRequired)
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

  val locusType = TLocus.schemaFromRG(referenceGenome)
  private val kType = TStruct("locus" -> locusType, "alleles" -> TArray(TString()))

  val fullType: MatrixType = MatrixType.fromParts(
    TStruct.empty(),
    colType = TStruct("s" -> TString()),
    colKey = Array("s"),
    rowType = kType ++ vaSignature,
    rowKey = Array("locus", "alleles"),
    entryType = genotypeSignature)

  val partitioner = if (partitionsJSON != null) {
    assert(inputs.length == 1)
    assert(minPartitions.isEmpty)
    assert(hConf.exists(inputs.head + ".tbi"))

    val pkType = TArray(TInterval(TStruct("locus" -> locusType)))
    val jv = JsonMethods.parse(partitionsJSON)
    val rangeBounds = JSONAnnotationImpex.importAnnotation(jv, pkType)

    new RVDPartitioner(
      Array("locus"),
      fullType.rowKeyStruct,
      rangeBounds.asInstanceOf[IndexedSeq[Interval]])
  } else
    null

  val fullRVDType: RVDType = RVDType(fullType.rvRowType.physicalType, fullType.rowKey)

  private lazy val lines = {
    HailContext.maybeGZipAsBGZip(gzAsBGZ) {
      ContextRDD.textFilesLines[RVDContext](sc, inputs, minPartitions, filterAndReplace)
    }
  }

  private lazy val coercer = RVD.makeCoercer(
    fullType.canonicalRVDType,
    1,
    parseLines(
      () => ()
    )((c, l, rvb) => ()
    )(lines,
      fullType.rowKeyStruct,
      referenceGenome,
      contigRecoding,
      arrayElementsRequired,
      skipInvalidLoci))

  def apply(mr: MatrixRead): MatrixValue = {
    val reader = new HtsjdkRecordReader(callFields, entryFloatType)
    val headerLinesBc = sc.broadcast(headerLines1)

    val requestedType = mr.typ
    assert(PruneDeadFields.isSupertype(requestedType, fullType))

    val localTyp = mr.typ
    val localInfoFlagFieldNames = infoFlagFieldNames

    val dropSamples = mr.dropCols
    val localSampleIDs: Array[String] = if (dropSamples) Array.empty[String] else sampleIDs

    val rvd = if (mr.dropRows)
      RVD.empty(sc, requestedType.canonicalRVDType)
    else
      coercer.coerce(requestedType.canonicalRVDType, parseLines { () =>
        new ParseLineContext(localTyp,
          localInfoFlagFieldNames,
          new BufferedLineIterator(headerLinesBc.value.iterator.buffered))
      } { (c, l, rvb) => LoadVCF.parseLine(reader, c, l, rvb, dropSamples) }(
        lines, requestedType.rvRowType, referenceGenome, contigRecoding, arrayElementsRequired, skipInvalidLoci
      ))

    MatrixValue(requestedType,
      BroadcastRow(Row.empty, requestedType.globalType, sc),
      BroadcastIndexedSeq(localSampleIDs.map(Annotation(_)), TArray(requestedType.colType), sc),
      rvd
    )
  }
}

object ImportVCFs {
  def pyApply(
    files: java.util.ArrayList[String],
    callFields: java.util.ArrayList[String],
    entryFloatType: String,
    rg: String,
    contigRecoding: java.util.Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean,
    gzAsBGZ: Boolean,
    forceGZ: Boolean,
    partitionsJSON: String,
    filter: String,
    find: String,
    replace: String
  ): Array[MatrixIR] = {
    val reader = VCFsReader(
      files.asScala.toArray,
      callFields.asScala.toSet,
      entryFloatType,
      Option(rg),
      Option(contigRecoding).map(_.asScala.toMap).getOrElse(Map.empty[String, String]),
      arrayElementsRequired,
      skipInvalidLoci,
      gzAsBGZ,
      forceGZ,
      TextInputFilterAndReplace(Option(find), Option(filter), Option(replace)),
      partitionsJSON)

    reader.read()
  }
}

case class VCFInfo(
  headerLines: Array[String],
  sampleIDs: Array[String],
  infoFlagFieldNames: Set[String],
  typ: MatrixType,
  partitions: Array[Partition])

case class VCFsReader(
  files: Array[String],
  callFields: Set[String],
  entryFloatType: String,
  rg: Option[String],
  contigRecoding: Map[String, String],
  arrayElementsRequired: Boolean,
  skipInvalidLoci: Boolean,
  gzAsBGZ: Boolean,
  forceGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  partitionsJSON: String) {

  private val hc = HailContext.get
  private val sc = hc.sc
  private val hConf = sc.hadoopConfiguration
  private val referenceGenome = rg.map(ReferenceGenome.getReference)

  referenceGenome.foreach(_.validateContigRemap(contigRecoding))

  private val locusType = TLocus.schemaFromRG(referenceGenome)
  private val rowKeyType = TStruct("locus" -> locusType)

  val partitioner: RVDPartitioner = {
    val pkType = TArray(TInterval(TStruct("locus" -> locusType)))
    val jv = JsonMethods.parse(partitionsJSON)
    val rangeBounds = JSONAnnotationImpex.importAnnotation(jv, pkType)

    new RVDPartitioner(
      Array("locus"),
      rowKeyType,
      rangeBounds.asInstanceOf[IndexedSeq[Interval]])
  }

  private val fileInfo = {
    val confBc = HailContext.hadoopConfBc

    val localLocusType = locusType
    val localReader = new HtsjdkRecordReader(callFields, entryFloatType)
    val localFiles = files
    val localArrayElementsRequired = arrayElementsRequired
    val localRangeBounds = partitioner.rangeBounds
    val localFilterAndReplace = filterAndReplace

    sc.parallelize(localFiles, localFiles.length).map { file =>
      val hConf = confBc.value.value
      val headerLines = getHeaderLines(hConf, file, localFilterAndReplace)
      val header = parseHeader(localReader, headerLines, arrayElementsRequired = localArrayElementsRequired)
      val VCFHeaderInfo(_, infoSignature, vaSignature, genotypeSignature, _, _, _, infoFlagFieldNames) = header

      val kType = TStruct("locus" -> localLocusType, "alleles" -> TArray(TString()))

      val typ = MatrixType.fromParts(
        TStruct.empty(),
        colType = TStruct("s" -> TString()),
        colKey = Array("s"),
        rowType = kType ++ vaSignature,
        rowKey = Array("locus"),
        entryType = genotypeSignature)

      val partitions = {
        val r = new TabixReader(file, hConf)
        localRangeBounds.zipWithIndex.map { case (b, i) =>
          if (!(b.includesStart && b.includesEnd))
            fatal("range bounds must be inclusive")

          val start = b.start.asInstanceOf[Row].getAs[Locus](0)
          val end = b.end.asInstanceOf[Row].getAs[Locus](0)
          if (start.contig != end.contig)
            fatal(s"partion spec must not cross contig boundaries, start: ${ start.contig } | end: ${ end.contig }")

          val contig = start.contig
          val startPos = start.position
          val endPos = end.position

          val tid = r.chr2tid(contig)
          val reg = r.queryPairs(tid, startPos - 1, endPos)

          val p: Partition = PartitionedVCFPartition(i, start.contig, start.position, end.position, reg)
          p
        }
          .toArray
      }

      VCFInfo(headerLines, header.sampleIds, infoFlagFieldNames, typ, partitions)
    }
      .collect()
  }

  def readFile(reader: HtsjdkRecordReader, file: String, i: Int): MatrixIR = {
    val VCFInfo(headerLines, sampleIDs, localInfoFlagFieldNames, typ, partitions) = fileInfo(i)

    val lines = ContextRDD.weaken[RVDContext](
      new PartitionedVCFRDD(sc, file, partitions)
        .map(l =>
          WithContext(l, Context(l, file, None))))

    val parsedLines = parseLines { () =>
      new ParseLineContext(typ,
        localInfoFlagFieldNames,
        new BufferedLineIterator(headerLines.iterator.buffered))
    } { (c, l, rvb) => LoadVCF.parseLine(reader, c, l, rvb) }(
      lines, typ.rvRowType, referenceGenome, contigRecoding, arrayElementsRequired, skipInvalidLoci
    )

    val rvd = RVD(typ.canonicalRVDType,
      partitioner,
      parsedLines)

    MatrixLiteral(
      MatrixValue(typ,
        BroadcastRow(Row.empty, typ.globalType, sc),
        BroadcastIndexedSeq(sampleIDs.map(Annotation(_)), TArray(typ.colType), sc),
        rvd))
  }

  def read(): Array[MatrixIR] = {
    val reader = new HtsjdkRecordReader(callFields, entryFloatType)
    files.zipWithIndex.map { case (file, i) =>
      readFile(reader, file, i)
    }
  }
}

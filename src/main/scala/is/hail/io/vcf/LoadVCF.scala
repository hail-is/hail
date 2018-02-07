package is.hail.io.vcf

import htsjdk.variant.vcf._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types._
import is.hail.expr._
import is.hail.io.{VCFAttributes, VCFMetadata}
import is.hail.rvd.OrderedRVD
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.RDD

import scala.annotation.switch
import scala.collection.JavaConversions._
import scala.language.implicitConversions
import scala.collection.mutable
import scala.io.Source

case class VCFHeaderInfo(sampleIds: Array[String], infoSignature: TStruct, vaSignature: TStruct, genotypeSignature: TStruct,
  filtersAttrs: VCFAttributes, infoAttrs: VCFAttributes, formatAttrs: VCFAttributes)

class VCFParseError(val msg: String, val pos: Int) extends RuntimeException(msg)

final class VCFLine(val line: String) {
  var pos: Int = 0

  val abs = new ArrayBuilder[String]
  val abi = new ArrayBuilder[Int]
  val abd = new ArrayBuilder[Double]

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
    var v = numericValue(line(pos))
    pos += 1
    while (!endField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v
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
  def parseAddVariant(rvb: RegionValueBuilder, gr: GenomeReference, contigRecoding: Map[String, String]): Boolean = {
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

    gr.checkLocus(recodedContig, start)

    skipField() // ID
    nextField()

    // REF
    val ref = parseString()
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
        rvb.addInt(
          Genotype.gtIndex(j, j))
      return
    }

    if (line(pos) != '|' && line(pos) != '/')
      parseError("parse error in call")
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
        parseError("ploidy > 2 not supported")
      else
        parseError("parse error in call")
    }

    // treat partially missing like missing
    if (mj || mk)
      rvb.setMissing()
    else {
      rvb.addInt(Genotype.gtIndexWithSwap(j, k))
    }
  }

  def parseFormatInt(): Int = {
    if (endFormatField())
      parseError("empty integer")
    var v = numericValue(line(pos))
    pos += 1
    while (!endFormatField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v
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
    var v = numericValue(line(pos))
    pos += 1
    while (!endFormatArrayField()) {
      v = v * 10 + numericValue(line(pos))
      pos += 1
    }
    v
  }

  def parseStringInFormatArray(): String = {
    val start = pos
    while (!endFormatField())
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseDoubleInFormatArray(): Double = {
    val s = parseStringInFormatArray()
    s.toDouble
  }

  def parseAddFormatArrayInt(rvb: RegionValueBuilder) {
    if (formatFieldMissing()) {
      rvb.setMissing()
      pos += 1
    } else {
      assert(abi.length == 0)

      abi += parseIntInFormatArray()
      while (!endFormatField()) {
        pos += 1 // comma
        abi += parseIntInFormatArray()
      }

      rvb.startArray(abi.length)
      var i = 0
      while (i < abi.length) {
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

      abs += parseStringInFormatArray()
      while (!endFormatField()) {
        pos += 1 // comma
        abs += parseStringInFormatArray()
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

      abd += parseDoubleInFormatArray()
      while (!endFormatField()) {
        pos += 1 // comma
        abd += parseDoubleInFormatArray()
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
      formatFields.map(gType.fieldIdx),
      (0 until gType.size)
        .filter(i => !formatFieldsSet.contains(gType.fields(i).name))
        .toArray)
  }
}

class FormatParser(
  gType: TStruct,
  formatFieldGIndex: Array[Int],
  missingGIndices: Array[Int]) {

  def parseAddField(l: VCFLine, rvb: RegionValueBuilder, i: Int) {
    val j = formatFieldGIndex(i)
    rvb.setFieldIndex(j)
    gType.fieldType(j) match {
      case TCall(_) =>
        l.parseAddCall(rvb)
      case TInt32(_) =>
        l.parseAddFormatInt(rvb)
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

  def setFieldMissing(rvb: RegionValueBuilder, i: Int) {
    rvb.setFieldIndex(formatFieldGIndex(i))
    rvb.setMissing()
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

class ParseLineContext(gType: TStruct, headerLines: BufferedLineIterator) {
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

  def globAllVCFs(arguments: Array[String], hConf: hadoop.conf.Configuration, forcegz: Boolean = false): Array[String] = {
    val inputs = hConf.globAll(arguments)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".vcf")
        && !input.endsWith(".vcf.bgz")) {
        if (input.endsWith(".vcf.gz")) {
          if (!forcegz)
            fatal(""".gz cannot be loaded in parallel. Is your file actually *block* gzipped?
                    |If your file is actually block gzipped (even though its extension is .gz),
                    |use force_bgz=True to ignore the file extension and treat this file as if
                    |it were a .bgz file. If you are sure that you want to load a non-block 
                    |gzipped using the very slow, non-parallel algorithm, use force=True.""".stripMargin)
        } else
          fatal(s"unknown input file type `$input', expect .vcf[.bgz]")
      }
    }
    inputs
  }

  def lineRef(s: String): String = {
    var i = 0
    var t = 0
    while (t < 3
      && i < s.length) {
      if (s(i) == '\t')
        t += 1
      i += 1
    }
    val start = i

    while (i < s.length
      && s(i) != '\t')
      i += 1
    val end = i

    s.substring(start, end)
  }

  def lineVariant(s: String): Variant = {
    val Array(contig, start, id, ref, alts, rest) = s.split("\t", 6)
    Variant(contig, start.toInt, ref, alts.split(","))
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

  def headerField(line: VCFCompoundHeaderLine, i: Int, callFields: Set[String], arrayElementsRequired: Boolean = false): (Field, (String, Map[String, String])) = {
    val id = line.getID
    val isCall = id == "GT" || callFields.contains(id)

    val baseType = (line.getType, isCall) match {
      case (VCFHeaderLineType.Integer, false) => TInt32()
      case (VCFHeaderLineType.Float, false) => TFloat64()
      case (VCFHeaderLineType.String, true) => TCall()
      case (VCFHeaderLineType.String, false) => TString()
      case (VCFHeaderLineType.Character, false) => TString()
      case (VCFHeaderLineType.Flag, false) => TBoolean()
      case (_, true) => fatal(s"Can only convert a header line with type `String' to a Call Type. Found `${ line.getType }'.")
    }

    val attrs = Map("Description" -> line.getDescription,
      "Number" -> headerNumberToString(line),
      "Type" -> headerTypeToString(line))

    if (line.isFixedCount &&
      (line.getCount == 1 ||
        (line.getType == VCFHeaderLineType.Flag && line.getCount == 0)))
      (Field(id, baseType, i), (id, attrs))
    else
      (Field(id, TArray(baseType.setRequired(arrayElementsRequired)), i), (id, attrs))
  }

  def headerSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    callFields: Set[String] = Set.empty[String], arrayElementsRequired: Boolean = false): (TStruct, VCFAttributes) = {
    val (fields, attrs) = lines
      .zipWithIndex
      .map { case (line, i) => headerField(line, i, callFields, arrayElementsRequired) }
      .unzip

    (TStruct(fields.toArray), attrs.toMap)
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
    val (infoSignature, infoAttrs) = headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val (gSignature, formatAttrs) = headerSignature(formatHeader, reader.callFields, arrayElementsRequired = arrayElementsRequired)

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

    VCFHeaderInfo(sampleIds, infoSignature, vaSignature, gSignature, filterAttrs, infoAttrs, formatAttrs)
  }

  def getHeaderLines[T](hConf: Configuration, file: String): Array[String] = hConf.readFile(file) { s =>
    Source.fromInputStream(s)
      .getLines()
      .takeWhile { line => line(0) == '#' }
      .toArray
  }

  // parses the Variant (key), leaves the rest to f
  def parseLines[C](makeContext: () => C)(f: (C, VCFLine, RegionValueBuilder) => Unit)(
    lines: RDD[WithContext[String]], t: Type, gr: GenomeReference, contigRecoding: Map[String, String]): RDD[RegionValue] = {
    lines.mapPartitions { it =>
      new Iterator[RegionValue] {
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        val context: C = makeContext()

        var present: Boolean = false

        def hasNext: Boolean = {
          while (!present && it.hasNext) {
            val lwc = it.next()
            val line = lwc.value
            try {
              val vcfLine = new VCFLine(line)
              region.clear()
              rvb.start(t)
              rvb.startStruct()
              present = vcfLine.parseAddVariant(rvb, gr, contigRecoding)
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

                fatal(s"${ source.locationString(pos) }: ${ e.msg }\n$prefix$excerpt$suffix\n$pad^\noffending line: @1\nsee the Hail log for the full offending line", line)
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

  def apply(hc: HailContext,
    reader: HtsjdkRecordReader,
    headerFile: Option[String],
    files: Array[String],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    gr: GenomeReference = GenomeReference.defaultReference,
    contigRecoding: Map[String, String] = Map.empty[String, String],
    arrayElementsRequired: Boolean = true): MatrixTable = {
    val sc = hc.sc
    val hConf = hc.hadoopConf

    val headerLines1 = getHeaderLines(hConf, headerFile.getOrElse(files.head))
    val header1 = parseHeader(reader, headerLines1, arrayElementsRequired = arrayElementsRequired)

    if (headerFile.isEmpty) {
      val confBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))
      val header1Bc = sc.broadcast(header1)

      sc.parallelize(files.tail, math.max(1, files.length - 1)).foreach { file =>
        val hConf = confBc.value.value
        val hd = parseHeader(reader, getHeaderLines(hConf, file))
        val hd1 = header1Bc.value

        if (hd1.sampleIds.length != hd.sampleIds.length) {
          fatal(
            s"""invalid sample ids: sample ids are different lengths.
             | ${ files(0) } has ${ hd1.sampleIds.length } ids and
             | ${ file } has ${ hd.sampleIds.length } ids.
           """.stripMargin)
        }

        hd1.sampleIds.iterator.zipAll(hd.sampleIds.iterator, None, None)
          .zipWithIndex.foreach { case ((s1, s2), i) =>
          if (s1 != s2) {
            fatal(
              s"""invalid sample ids: expected sample ids to be identical for all inputs. Found different sample ids at position $i.
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
      warn("Loading user-provided header file. The number of samples, sample IDs, variant annotation schema and genotype schema were not checked for agreement with input data.")
    }

    val VCFHeaderInfo(sampleIdsHeader, infoSignature, vaSignature, genotypeSignature, _, _, _) = header1

    val sampleIds: Array[String] = if (dropSamples) Array.empty else sampleIdsHeader
    val localNSamples = sampleIds.length

    LoadVCF.warnDuplicates(sampleIds)

    val infoSignatureBc = sc.broadcast(infoSignature)

    val headerLinesBc = sc.broadcast(headerLines1)

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions))

    val matrixType: MatrixType = MatrixType.fromParts(
      TStruct.empty(true),
      colType = TStruct("s" -> TString()),
      colKey = Array("s"),
      rowType = TStruct("locus" -> TLocus(gr), "alleles" -> TArray(TString())) ++ vaSignature,
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = genotypeSignature)

    val kType = matrixType.orderedRVType.kType
    val rowType = matrixType.rvRowType

    // nothing after the key
    val justVariants = parseLines(() => ())((c, l, rvb) => ())(lines, kType, gr, contigRecoding)

    val rdd = OrderedRVD(
      matrixType.orderedRVType,
      parseLines { () =>
        new ParseLineContext(genotypeSignature, new BufferedLineIterator(headerLinesBc.value.iterator.buffered))
      } { (c, l, rvb) =>
        val vc = c.codec.decode(l.line)
        reader.readVariantInfo(vc, rvb, infoSignatureBc.value)

        rvb.startArray(localNSamples) // gs
        if (localNSamples > 0) {
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
          while (i < localNSamples) {
            l.nextField()
            fp.parse(l, rvb)
            i += 1
          }
        }
        rvb.endArray()
      }(lines, rowType, gr, contigRecoding),
      Some(justVariants), None)

    new MatrixTable(hc,
      matrixType,
      MatrixLocalValue(Annotation.empty,
        sampleIds.map(x => Annotation(x))),
      rdd)
  }

  def parseHeaderMetadata(hc: HailContext, reader: HtsjdkRecordReader, headerFile: String): VCFMetadata = {
    val hConf = hc.hadoopConf
    val headerLines = getHeaderLines(hConf, headerFile)
    val VCFHeaderInfo(_, _, _, _, filtersAttrs, infoAttrs, formatAttrs) = parseHeader(reader, headerLines)

    Map("filter" -> filtersAttrs, "info" -> infoAttrs, "format" -> formatAttrs)
  }
}

package is.hail.io.vcf

import java.util

import htsjdk.variant.vcf._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{TStruct, _}
import is.hail.sparkextras.OrderedRDD2
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.RDD
import scala.collection.JavaConversions._
import scala.language.implicitConversions
import scala.collection.mutable
import scala.io.Source


case class VCFHeaderInfo(sampleIds: Array[String], infoSignature: TStruct, vaSignature: TStruct, genotypeSignature: TStruct, canonicalFlags: Int)

class VCFInput(var s: String = null, var pos: Int = 0) {
  def length: Int = s.length

  def apply(i: Int): Char = s(i)

  def set(newS: String): Unit = set(newS, 0)

  def set(newS: String, newPos: Int) {
    s = newS
    pos = newPos
  }

  def nonEmpty: Boolean = pos < s.length

  def head: Char = s(pos)

  def next(): Char = {
    val c = s(pos)
    pos += 1
    c
  }
}

object FormatParser {
  def formatArrayLength(in: String, start: Int, end: Int): Int = {
    var length = 1
    var p = start
    while (p < end && (in(p) != '\t' && in(p) != ':')) {
      if (in(p) == ',')
        length += 1
      p += 1
    }
    length
  }

  def formatAddCall(in: String, start: Int, end: Int, rvb: RegionValueBuilder) {
    var k = start
    if (start == end) {
      rvb.setMissing()
      return
    } else if (in(k) == '.') {
      rvb.setMissing()
      return
    }

    var c = in(k)
    if (c < '0' || c > '9')
      fatal("parse error in call")
    k += 1
    var i = 0
    do {
      i = i * 10 + (c - '0')
    } while (k < end && {
      c = in(k)
      k += 1
      c >= '0' && c <= '9'
    })

    val gt =
      if (c == '|' || c == '/') {
        assert(k < end)
        c = in(k)
        k += 1
        assert(c >= '0' && c <= '9')
        var j = 0
        do {
          j = j * 10 + (c - '0')
        } while (k < end && {
          c = in(k)
          k += 1
          c >= '0' && c <= '9'
        })

        Genotype.gtIndexWithSwap(i, j)
      } else {
        if (k != end)
          fatal("parse error in call")
        Genotype.gtIndex(i, i)
      }
    rvb.addInt(gt)
  }

  def formatAddInt(in: String, start: Int, end: Int, rvb: RegionValueBuilder) {
    assert(start != end)
    if (in(start) == '.') {
      rvb.setMissing()
      return
    }

    var i = start
    var x = 0
    while (i < end) {
      var c = in(i)
      assert(c >= 0 && c <= '9')
      x = x * 10 + (c - '0')
      i += 1
    }
    rvb.addInt(x)
  }

  def formatAddArrayInt(in: String, start: Int, end: Int, rvb: RegionValueBuilder) {
    if (start == end) {
      rvb.startArray(0)
      rvb.end()
      return
    } else if (in(start) == '.') {
      rvb.setMissing()
      return
    }

    val length = formatArrayLength(in, start, end)
    rvb.startArray(length)

    var i = start
    var x = 0
    do {
      var c = in(i)
      if (c == ',') {
        rvb.addInt(x)
        x = 0
      } else
        x = x * 10 + (c - '0')
      i += 1
    } while (i < end)
    rvb.addInt(x)

    rvb.endArray()
  }
}

class FormatParser(nGenotypeFields: Int,
  formatGenotypeIndex: Array[Int],
  genotypeFieldTypes: Array[Type]) {
  val nFormatFields: Int = formatGenotypeIndex.length

  val fieldStart = new Array[Int](nGenotypeFields)
  val fieldEnd = new Array[Int](nGenotypeFields)

  def calculateFieldPositions(in: String, start: Int): Int = {
    util.Arrays.fill(fieldStart, -1)

    var p = start
    var f = 0
    var i = formatGenotypeIndex(f)
    fieldStart(i) = p
    while (p < in.length && in(p) != '\t') {
      if (in(p) == ':') {
        fieldEnd(i) = p
        f += 1
        i = formatGenotypeIndex(f)
        fieldStart(i) = p + 1
      }
      p += 1
    }
    fieldEnd(i) = p
    p
  }

  def parseFormat(in: String, start: Int, rvb: RegionValueBuilder): Int = {
    val end = calculateFieldPositions(in, start)

    rvb.startStruct()
    var i = 0
    while (i < nGenotypeFields) {
      val start = fieldStart(i)
      if (start == -1)
        rvb.setMissing()
      else {
        val end = fieldEnd(i)
        genotypeFieldTypes(i) match {
          case TCall =>
            FormatParser.formatAddCall(in, start, end, rvb)
          case TInt32 =>
            FormatParser.formatAddInt(in, start, end, rvb)
          case TArray(TInt32) =>
            FormatParser.formatAddArrayInt(in, start, end, rvb)
          case TString =>
            rvb.addString(in.substring(start, end))
          case TFloat64 =>
            rvb.addDouble(in.substring(start, end).toDouble)
        }
      }

      i += 1
    }
    rvb.endStruct()

    end
  }
}

class Position {
  var start: Int = _
  var end: Int = _
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
            fatal(".gz cannot be loaded in parallel, use .bgz or force=True override")
        } else
          fatal(s"unknown input file type `$input', expect .vcf[.bgz]")
      }
    }
    inputs
  }

  def columnPositions(in: String,
    columnStart: Array[Int],
    columnEnd: Array[Int]): Int = {
    val n = columnStart.length
    var k = 0
    columnStart(0) = 0
    var c = 0
    while (true) {
      if (k == in.length || in(k) == '\t') {
        columnEnd(c) = k
        c += 1
        if (k < in.length && c < n)
          columnStart(c) = k + 1
        else
          return c
      }
      k += 1
    }
    -1
  }

  def refIsGood(in: String, start: Int, end: Int): Boolean = {
    var i = start
    while (i < end) {
      val c = in(i)
      if (c != 'A' && c != 'C' && c != 'G' && c != 'T' && c == 'N')
        return false
      i += 1
    }
    true
  }

  def containsSymblicAllele(in: String, start: Int, end: Int): Boolean = {
    var i = start
    while (i < end) {
      val c = in(i)
      if (c == '<')
        return true
      i += 1
    }
    false
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

  def headerField(line: VCFCompoundHeaderLine, i: Int, callFields: Set[String]): Field = {
    val id = line.getID
    val isCall = id == "GT" || callFields.contains(id)

    val baseType = (line.getType, isCall) match {
      case (VCFHeaderLineType.Integer, false) => TInt32
      case (VCFHeaderLineType.Float, false) => TFloat64
      case (VCFHeaderLineType.String, true) => TCall
      case (VCFHeaderLineType.String, false) => TString
      case (VCFHeaderLineType.Character, false) => TString
      case (VCFHeaderLineType.Flag, false) => TBoolean
      case (_, true) => fatal(s"Can only convert a header line with type `String' to a Call Type. Found `${ line.getType }'.")
    }

    val attrs = Map("Description" -> line.getDescription,
      "Number" -> headerNumberToString(line),
      "Type" -> headerTypeToString(line))

    if (line.isFixedCount &&
      (line.getCount == 1 ||
        (line.getType == VCFHeaderLineType.Flag && line.getCount == 0)))
      Field(id, baseType, i, attrs)
    else
      Field(id, TArray(baseType), i, attrs)
  }

  def headerSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    callFields: Set[String] = Set.empty[String]): TStruct = {
    TStruct(lines
      .zipWithIndex
      .map { case (line, i) => headerField(line, i, callFields) }
      .toArray)
  }

  def formatHeaderSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    callFields: Set[String] = Set.empty[String]): (TStruct, Int) = {
    val canonicalFields = Array(
      "GT" -> TCall,
      "AD" -> TArray(TInt32),
      "DP" -> TInt32,
      "GQ" -> TInt32,
      "PL" -> TArray(TInt32))

    val raw = headerSignature(lines, callFields)

    var canonicalFlags = 0
    var i = 0
    val done = mutable.Set[Int]()
    val fb = new ArrayBuilder[Field]()
    canonicalFields.zipWithIndex.foreach { case ((id, t), j) =>
      if (raw.hasField(id)) {
        val f = raw.field(id)
        if (f.typ == t) {
          done += f.index
          fb += Field(f.name, f.typ, i, f.attrs)
          canonicalFlags |= (1 << j)
          i += 1
        }
      }
    }

    raw.fields.foreach { f =>
      if (!done.contains(f.index)) {
        fb += Field(f.name, f.typ, i, f.attrs)
        i += 1
      }
    }

    (TStruct(fb.result()), canonicalFlags)
  }

  def parseInt(in: VCFInput): Int = {
    var x = 0
    do {
      val c = in.next()
      assert(c >= '0' && c <= '9')
      x = x * 10 + (c - '0')
    } while (in.nonEmpty && in.head != '\t')
    x
  }

  def parseString(in: VCFInput, sb: StringBuilder): String = {
    sb.clear()
    while (in.nonEmpty && in.head != '\t') {
      sb += in.next()
    }
    sb.result()
  }

  def parseAddString(in: VCFInput, sb: StringBuilder, rvb: RegionValueBuilder) {
    rvb.addString(parseString(in, sb))
  }

  def parseArrayLength(in: VCFInput): Int = {
    var length = 1
    var p = in.pos
    while (p < in.length && in(p) != '\t') {
      if (in(p) == ',')
        length += 1
      p += 1
    }
    length
  }

  def parseAddAlt(ref: String, in: VCFInput, sb: StringBuilder, rvb: RegionValueBuilder) {
    val length = parseArrayLength(in)
    rvb.startArray(length)

    sb.clear()
    while (in.nonEmpty && in.head != '\t') {
      val c = in.next()
      if (c == ',') {
        rvb.startStruct() // altAllele
        rvb.addString(ref)
        rvb.addString(sb.result())
        rvb.endStruct()
        sb.clear()
      } else
        sb += c
    }
    rvb.startStruct() // altAllele
    rvb.addString(ref)
    rvb.addString(sb.result())
    rvb.endStruct()

    rvb.endArray()
  }

  def skipField(in: VCFInput) {
    while (in.nonEmpty && in.head != '\t')
      in.next()
  }

  def parseHeader(reader: HtsjdkRecordReader, lines: Array[String]): VCFHeaderInfo = {

    val codec = new htsjdk.variant.vcf.VCFCodec()
    val header = codec.readHeader(new BufferedLineIterator(lines.iterator.buffered))
      .getHeaderValue
      .asInstanceOf[htsjdk.variant.vcf.VCFHeader]

    // FIXME apply descriptions when HTSJDK is fixed to expose filter descriptions
    val filters: Map[String, String] = header
      .getFilterLines
      .toList
      // (ID, description)
      .map(line => (line.getID, ""))
      .toMap

    val infoHeader = header.getInfoHeaderLines
    val infoSignature = headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val (gSignature, canonicalFlags) = formatHeaderSignature(formatHeader, reader.callFields)

    val vaSignature = TStruct(Array(
      Field("rsid", TString, 0),
      Field("qual", TFloat64, 1),
      Field("filters", TSet(TString), 2, filters),
      Field("info", infoSignature, 3)))

    val headerLine = lines.last
    if (!(headerLine(0) == '#' && headerLine(1) != '#'))
      fatal(
        s"""corrupt VCF: expected final header line of format `#CHROM\tPOS\tID...'
           |  found: @1""".stripMargin, headerLine)

    val sampleIds: Array[String] = headerLine.split("\t").drop(9)

    VCFHeaderInfo(sampleIds, infoSignature, vaSignature, gSignature, canonicalFlags)
  }

  def getHeaderLines[T](hConf: Configuration, file: String): Array[String] = hConf.readFile(file) { s =>
    Source.fromInputStream(s)
      .getLines()
      .takeWhile { line => line(0) == '#' }
      .toArray
  }

  def apply(hc: HailContext,
    reader: HtsjdkRecordReader,
    file1: String,
    files: Array[String],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    gr: GenomeReference = GenomeReference.GRCh37): VariantSampleMatrix[Locus, Variant, Annotation] = {
    val sc = hc.sc
    val hConf = hc.hadoopConf

    val headerLines1 = getHeaderLines(hConf, file1)
    val header1 = parseHeader(reader, headerLines1)
    val header1Bc = sc.broadcast(header1)

    val confBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))

    sc.parallelize(files.tail, math.max(1, files.length - 1)).foreach { file =>
      val hConf = confBc.value.value
      val hd = parseHeader(reader, getHeaderLines(hConf, file))
      val hd1 = header1Bc.value

      hd1.sampleIds.iterator.zip(hd.sampleIds.iterator)
        .zipWithIndex.dropWhile { case ((s1, s2), i) => s1 == s2 }.toArray.headOption match {
        case Some(((s1, s2), i)) => fatal(
          s"""invalid sample ids: expected sample ids to be identical for all inputs. Found different sample ids at position $i.
             |    ${ files(0) }: $s1
             |    $file: $s2""".stripMargin)
        case None =>
      }

      if (hd1.genotypeSignature != hd.genotypeSignature)
        fatal(
          s"""invalid genotype signature: expected signatures to be identical for all inputs.
             |   ${ files(0) }: ${ hd1.genotypeSignature.toPrettyString(compact = true, printAttrs = true) }
             |   $file: ${ hd.genotypeSignature.toPrettyString(compact = true, printAttrs = true) }""".stripMargin)

      if (hd1.vaSignature != hd.vaSignature)
        fatal(
          s"""invalid variant annotation signature: expected signatures to be identical for all inputs.
             |   ${ files(0) }: ${ hd1.vaSignature.toPrettyString(compact = true, printAttrs = true) }
             |   $file: ${ hd.vaSignature.toPrettyString(compact = true, printAttrs = true) }""".stripMargin)
    }

    val VCFHeaderInfo(sampleIdsHeader, infoSignature, vaSignature, genotypeSignature, canonicalFlags) = header1

    val sampleIds: Array[String] =
      if (dropSamples)
        Array.empty
      else
        sampleIdsHeader

    val nSamples = sampleIds.length

    LoadVCF.warnDuplicates(sampleIds)

    val infoSignatureBc = sc.broadcast(infoSignature)
    val genotypeSignatureBc = sc.broadcast(genotypeSignature)

    val headerLinesBc = sc.broadcast(headerLines1)

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions))

    val fullKeyType = TStruct(
      "pk" -> TLocus(gr),
      "k" -> TVariant(gr))

    val variants: RDD[RegionValue] = lines.mapPartitions { it =>
      val codec = new htsjdk.variant.vcf.VCFCodec()
      codec.readHeader(new BufferedLineIterator(headerLinesBc.value.iterator.buffered))

      val in = new VCFInput()
      val sb = new StringBuilder()
      val region = MemoryBuffer()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      val columnStart = new Array[Int](5)
      val columnEnd = new Array[Int](5)

      new Iterator[RegionValue] {
        var present = false

        def advance() {
          while (!present && it.hasNext) {
            it.next().foreach { line =>
              if (line.nonEmpty && line(0) != '#') {
                val nColumns = columnPositions(line, columnStart, columnEnd)
                assert(nColumns == 5)

                if (refIsGood(line, columnStart(3), columnEnd(3))
                  || !containsSymblicAllele(line, columnStart(4), columnEnd(4))) {
                  val vc = codec.decode(line)
                  in.set(line)

                  val contig = parseString(in, sb)
                  in.next()
                  val start = parseInt(in)
                  in.next()
                  skipField(in)
                  in.next()
                  val ref = parseString(in, sb)
                  in.next()

                  region.clear()
                  rvb.start(fullKeyType)
                  rvb.startStruct() // fk

                  rvb.startStruct() // pk: Locus
                  rvb.addString(contig)
                  rvb.addInt(start)
                  rvb.endStruct()

                  rvb.startStruct() // k: Variant
                  rvb.addString(contig)
                  rvb.addInt(start)
                  rvb.addString(ref)
                  parseAddAlt(ref, in, sb, rvb) // alts
                  rvb.endStruct()

                  rvb.endStruct() // fk
                  rv.setOffset(rvb.end())

                  present = true
                }
              }
            }
          }
        }

        def hasNext: Boolean = {
          if (!present)
            advance()
          present
        }

        def next(): RegionValue = {
          hasNext
          assert(present)
          present = false
          rv
        }
      }
    }

    // FIXME
    val noMulti = false
    // val noMulti = justVariants.forall(_.nAlleles == 2)

    if (noMulti)
      info("No multiallelics detected.")
    else
      info("Multiallelic variants detected. Some methods require splitting or filtering multiallelics first.")

    val rowType = TStruct(
      "pk" -> TLocus(gr),
      "v" -> TVariant(gr),
      "va" -> header1.vaSignature,
      "gs" -> TArray(genotypeSignature))

    val rowTypeTreeBc = BroadcastTypeTree(sc, rowType)

    val rdd = lines
      .mapPartitions { it =>
        val codec = new htsjdk.variant.vcf.VCFCodec()
        codec.readHeader(new BufferedLineIterator(headerLinesBc.value.iterator.buffered))

        val columnStart = new Array[Int](9)
        val columnEnd = new Array[Int](9)

        val region = MemoryBuffer()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region, 0)

        val formatParsers = mutable.Map[String, FormatParser]()

        def getFormatParser(format: String): FormatParser = {
          if (formatParsers.contains(format))
            formatParsers(format)
          else {
            val formatFields = format.split(':')
            val parser = new FormatParser(genotypeSignature.size,
              formatFields.map(genotypeSignature.fieldIdx),
              genotypeSignature.fields.map(_.typ).toArray)
            formatParsers += format -> parser
            parser
          }
        }

        new Iterator[RegionValue] {
          var present = false

          def advance() {
            while (!present && it.hasNext) {
              it.next().foreach { line =>
                if (line.nonEmpty && line(0) != '#') {
                  val nColumns = columnPositions(line, columnStart, columnEnd)
                  assert(nColumns == 8 || nColumns == 9)

                  if (refIsGood(line, columnStart(3), columnEnd(3))
                    && !containsSymblicAllele(line, columnStart(4), columnEnd(4))) {
                    val vc = codec.decode(line)
                    assert(!vc.isSymbolic)

                    val formatParser: FormatParser =
                      if (nColumns == 9)
                        getFormatParser(line.substring(columnStart(8), columnEnd(8)))
                      else
                        null

                    region.clear()
                    rvb.start(rowType)
                    rvb.startStruct()

                    reader.readVariantInfo(vc, rvb, infoSignature)

                    rvb.startArray(nSamples) // gs

                    if (nSamples > 0) {
                      var k = columnEnd(8)
                      while (k < line.length) {
                        val c = line(k)
                        assert(c == '\t')
                        k += 1
                        k = formatParser.parseFormat(line, k, rvb)
                      }
                    }

                    rvb.endArray()
                    rvb.endStruct() // row
                    rv.setOffset(rvb.end())

                    present = true
                  }
                }
              }
            }
          }

          def hasNext: Boolean = {
            if (!present)
              advance()
            present
          }

          def next(): RegionValue = {
            if (!present)
              advance()
            assert(present)
            present = false
            rv
          }
        }
      }

    val ordd = OrderedRDD2("pk", "v", rowType, rdd, Some(variants), None)

    new VariantSampleMatrix(hc, VSMMetadata(
      TString,
      TStruct.empty,
      TVariant(gr),
      vaSignature,
      TStruct.empty,
      genotypeSignature,
      wasSplit = noMulti),
      VSMLocalValue(Annotation.empty,
        sampleIds,
        Annotation.emptyIndexedSeq(sampleIds.length)),
      ordd)
  }
}

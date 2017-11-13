package is.hail.io.vcf

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

import scala.annotation.switch
import scala.collection.JavaConversions._
import scala.language.implicitConversions
import scala.collection.mutable
import scala.io.Source

case class VCFHeaderInfo(sampleIds: Array[String], infoSignature: TStruct, vaSignature: TStruct, genotypeSignature: TStruct, canonicalFlags: Int)

object VCFLine {
  def numericValue(c: Char): Int = {
    if (c < '0' || c > '9')
      fatal(s"invalid character '$c' in integer literal")
    c - '0'
  }

  def invalidRef(ref: String): Boolean = {
    var i = 0
    while (i < ref.length) {
      val c = ref(i)
      (c: @switch) match {
        case 'A' | 'C' | 'G' | 'T' | 'N' |
             'a' | 'c' | 'g' | 't' | 'n' =>
        case _ =>
          return true
      }
      i += 1
    }
    false
  }

  def symbolicAlt(alt: String): Boolean = {
    assert(alt.length > 0)
    val f = alt(0)
    val l = alt(alt.length - 1)
    if (f == '.' || l == '.' || (f == '<' && l == '>'))
      return true

    var i = 0
    while (i < alt.length) {
      val c = alt(i)
      if (c == '[' || c == ']')
        return true
      i += 1
    }
    false
  }
}

final class VCFLine(line: String) {
  var pos: Int = 0

  val abs = new ArrayBuilder[String]

  def parseString(): String = {
    val start = pos
    while (pos < line.length && line(pos) != '\t')
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  def parseInt(): Int = {
    var c: Char = 0

    if (!(pos < line.length && line(pos) != '\t'))
      fatal("empty integer field")
    var v = VCFLine.numericValue(line(pos))
    pos += 1
    while (pos < line.length && line(pos) != '\t') {
      v = v * 10 + VCFLine.numericValue(line(pos))
      pos += 1
    }
    v
  }

  def skipField(): Unit = {
    while (pos < line.length && line(pos) != '\t')
      pos += 1
  }

  def parseStringInArray(): String = {
    val start = pos
    while (pos < line.length && line(pos) != '\t' && line(pos) != ',')
      pos += 1
    val end = pos
    line.substring(start, end)
  }

  // leaves result in abs
  def parseAltAlleles(): Unit = {
    assert(abs.size == 0)

    // . means no alternate alleles
    if (pos < line.length
      && line(pos) == '.'
      && (pos + 1 == line.length || line(pos + 1) == '\t'))
      return

    abs += parseStringInArray()
    while (pos < line.length && line(pos) != '\t') {
      pos += 1 // comma
      abs += parseStringInArray()
    }
  }

  def nextField(): Unit = {
    if (pos == line.length)
      fatal("unexpected end of line")
    pos += 1 // tab
  }

  // return false if it should be filtered
  def parseAddVariant(rvb: RegionValueBuilder): Boolean = {
    assert(pos == 0)

    if (line.isEmpty || line(0) == '#')
      return false

    // CHROM (contig)
    val contig = parseString()
    nextField()

    // POS (start)
    val start = parseInt()
    nextField()

    skipField() // ID
    nextField()

    // REF
    val ref = parseString()
    if (VCFLine.invalidRef(ref)) {
      warn(s"skipping variant with invalid REF field: $ref")
      return false
    }
    nextField()

    // ALT
    parseAltAlleles()

    var i = 0
    while (i < abs.length) {
      val alt = abs(i)
      if (alt.isEmpty)
        fatal("empty alternate allele")
      if (VCFLine.symbolicAlt(alt)) {
        warn(s"skipping variant with symbolic alternate allele in ALT field: $alt")
        return false
      }
      i += 1
    }

    rvb.startStruct() // pk: Locus
    rvb.addString(contig)
    rvb.addInt(start)
    rvb.endStruct()

    rvb.startStruct() // v: Variant
    rvb.addString(contig)
    rvb.addInt(start)
    rvb.addString(ref)
    rvb.startArray(abs.length)
    i = 0
    while (i < abs.length) {
      rvb.startStruct()
      rvb.addString(ref)
      rvb.addString(abs(i))
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct() // v

    abs.clear()

    true
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
            fatal(".gz cannot be loaded in parallel, use .bgz or force=True override")
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

  def headerField(line: VCFCompoundHeaderLine, i: Int, callFields: Set[String], arrayElementsRequired: Boolean = false): Field = {
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
      Field(id, baseType, i, attrs)
    else
      Field(id, TArray(baseType.setRequired(arrayElementsRequired)), i, attrs)
  }

  def headerSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    callFields: Set[String] = Set.empty[String], arrayElementsRequired: Boolean = false): TStruct = {
    TStruct(lines
      .zipWithIndex
      .map { case (line, i) => headerField(line, i, callFields, arrayElementsRequired) }
      .toArray)
  }

  def formatHeaderSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    callFields: Set[String] = Set.empty[String], arrayElementsRequired: Boolean = true): (TStruct, Int) = {
    val canonicalFields = Array(
      "GT" -> TCall(),
      "AD" -> TArray(TInt32(arrayElementsRequired)),
      "DP" -> TInt32(),
      "GQ" -> TInt32(),
      "PL" -> TArray(TInt32(arrayElementsRequired)))

    val raw = headerSignature(lines, callFields, arrayElementsRequired)

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

  def parseHeader(reader: HtsjdkRecordReader, lines: Array[String], arrayElementsRequired: Boolean = true): VCFHeaderInfo = {

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
    val (gSignature, canonicalFlags) = formatHeaderSignature(formatHeader, reader.callFields, arrayElementsRequired = arrayElementsRequired)

    val vaSignature = TStruct(Array(
      Field("rsid", TString(), 0),
      Field("qual", TFloat64(), 1),
      Field("filters", TSet(TString()), 2, filters),
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

  // parses the Variant (key), leaves the rest to f
  def parseLines[C](makeContext: () => C)(f: (C, String, RegionValueBuilder) => Unit)(
    lines: RDD[WithContext[String]], t: Type): RDD[RegionValue] = {
    lines.mapPartitions { it =>
      new Iterator[RegionValue] {
        val region = MemoryBuffer()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        val context: C = makeContext()

        var present: Boolean = false

        def hasNext: Boolean = {
          while (!present && it.hasNext) {
            present = it.next().map { l =>
              val vcfLine = new VCFLine(l)
              region.clear()
              rvb.start(t)
              rvb.startStruct()
              val b = vcfLine.parseAddVariant(rvb)
              if (b) {
                f(context, l, rvb)

                rvb.endStruct()
                rv.setOffset(rvb.end())
              } else
                rvb.clear()
              b
            }.value
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
    file1: String,
    files: Array[String],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    gr: GenomeReference = GenomeReference.GRCh37,
    arrayElementsRequired: Boolean = true): VariantSampleMatrix[Locus, Variant, Annotation] = {
    val sc = hc.sc
    val hConf = hc.hadoopConf

    val headerLines1 = getHeaderLines(hConf, file1)
    val header1 = parseHeader(reader, headerLines1, arrayElementsRequired = arrayElementsRequired)
    val header1Bc = sc.broadcast(header1)

    val confBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))

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
             |   ${ files(0) }: ${ hd1.genotypeSignature.toPrettyString(compact = true, printAttrs = true) }
             |   $file: ${ hd.genotypeSignature.toPrettyString(compact = true, printAttrs = true) }""".stripMargin)

      if (hd1.vaSignature != hd.vaSignature)
        fatal(
          s"""invalid variant annotation signature: expected signatures to be identical for all inputs.
             |   ${ files(0) }: ${ hd1.vaSignature.toPrettyString(compact = true, printAttrs = true) }
             |   $file: ${ hd.vaSignature.toPrettyString(compact = true, printAttrs = true) }""".stripMargin)
    }

    val VCFHeaderInfo(sampleIdsHeader, infoSignature, vaSignature, genotypeSignature, canonicalFlags) = header1

    val sampleIds: Array[String] = if (dropSamples) Array.empty else sampleIdsHeader

    LoadVCF.warnDuplicates(sampleIds)

    val infoSignatureBc = sc.broadcast(infoSignature)
    val genotypeSignatureBc = sc.broadcast(genotypeSignature)

    val headerLinesBc = sc.broadcast(headerLines1)

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions))

    val gr = GenomeReference.GRCh37
    val vsmMetadata = VSMMetadata(
      TString(),
      TStruct.empty,
      TVariant(gr),
      vaSignature,
      TStruct.empty,
      genotypeSignature)
    val matrixType = MatrixType(vsmMetadata)

    val locusType = matrixType.locusType
    val vType = matrixType.vType
    val kType = matrixType.kType
    val rowType = matrixType.rowType

    // nothing after the key
    val justVariants = parseLines(() => ())((c, l, rvb) => ())(lines, kType)

    val rdd = OrderedRDD2(
      matrixType.orderedRDD2Type,
      parseLines { () =>
        val codec = new htsjdk.variant.vcf.VCFCodec()
        codec.readHeader(new BufferedLineIterator(headerLinesBc.value.iterator.buffered))
        codec
      } { (c, l, rvb) =>
        val vc = c.decode(l)
        reader.readRecord(vc, rvb, infoSignatureBc.value, genotypeSignatureBc.value, dropSamples, canonicalFlags)
      } (lines, rowType),
      Some(justVariants), None)

    new VariantSampleMatrix(hc,
      vsmMetadata,
      VSMLocalValue(Annotation.empty,
        sampleIds,
        Annotation.emptyIndexedSeq(sampleIds.length)),
      rdd)
  }
}

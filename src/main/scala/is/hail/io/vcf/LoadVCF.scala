package is.hail.io.vcf

import htsjdk.tribble.TribbleException
import htsjdk.variant.vcf._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{TStruct, _}
import is.hail.sparkextras.OrderedRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop
import org.apache.spark.Accumulable
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source

case class VCFSettings(storeGQ: Boolean = false,
  skipGenotypes: Boolean = false,
  ppAsPL: Boolean = false,
  skipBadAD: Boolean = false)

case class LoadVCFResult[T](rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])], sampleIds: Array[String], vaSignature: TStruct, gSignature: Type)

object VCFReport {
  val GTPLMismatch = 1
  val ADDPMismatch = 2
  val ODMissingAD = 3
  val ADODDPPMismatch = 4
  val GQPLMismatch = 5
  val GQMissingPL = 6
  val RefNonACGTN = 7
  val Symbolic = 8
  val ADInvalidNumber = 9

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def isVariant(id: Int): Boolean = id == RefNonACGTN || id == Symbolic

  def isGenotype(id: Int): Boolean = !isVariant(id)

  def warningMessage(id: Int, count: Int): String = {
    val desc = id match {
      case GTPLMismatch => "PL(GT) != 0"
      case ADDPMismatch => "sum(AD) > DP"
      case ODMissingAD => "OD present but AD missing"
      case ADODDPPMismatch => "DP != sum(AD) + OD"
      case GQPLMismatch => "GQ != difference of two smallest PL entries"
      case GQMissingPL => "GQ present but PL missing"
      case RefNonACGTN => "REF contains non-ACGTN"
      case Symbolic => "Variant is symbolic"
      case ADInvalidNumber => "AD array contained the wrong number of elements"
    }
    s"$count ${ plural(count, "time") }: $desc"
  }

  def report() {
    val sb = new StringBuilder()
    for ((file, m) <- accumulators) {
      sb.clear()

      sb.append(s"while importing:\n    $file")

      val variantWarnings = m.value.filter { case (k, v) => isVariant(k) }
      val nVariantsFiltered = variantWarnings.values.sum
      if (nVariantsFiltered > 0) {
        sb.append(s"\n  filtered $nVariantsFiltered variants:")
        variantWarnings.foreach { case (id, n) =>
          if (n > 0) {
            sb.append("\n    ")
            sb.append(warningMessage(id, n))
          }
        }
        warn(sb.result())
      }

      val genotypeWarnings = m.value.filter { case (k, v) => isGenotype(k) }
      val nGenotypesFiltered = genotypeWarnings.values.sum
      if (nGenotypesFiltered > 0) {
        sb.append(s"\n  filtered $nGenotypesFiltered genotypes:")
        genotypeWarnings.foreach { case (id, n) =>
          if (n > 0) {
            sb.append("\n    ")
            sb.append(warningMessage(id, n))
          }
        }
      }

      if (nVariantsFiltered == 0 && nGenotypesFiltered == 0) {
        sb.append("  import clean")
        info(sb.result())
      } else
        warn(sb.result())
    }
  }
}


object LoadVCF {

  def globAllVCFs(arguments: Array[String], hConf: hadoop.conf.Configuration, forcegz: Boolean = false): Array[String] = {
    val inputs = hConf.globAll(arguments)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".vcf")
        && !input.endsWith(".vcf.bgz")) {
        if (input.endsWith(".vcf.gz")) {
          if (!forcegz)
            fatal(".gz cannot be loaded in parallel, use .bgz or -f override")
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

  def headerField(line: VCFCompoundHeaderLine, i: Int): Field = {
    val baseType = line.getType match {
      case VCFHeaderLineType.Integer => TInt
      case VCFHeaderLineType.Float => TDouble
      case VCFHeaderLineType.String => TString
      case VCFHeaderLineType.Character => TChar
      case VCFHeaderLineType.Flag => TBoolean
    }

    val attrs = Map("Description" -> line.getDescription,
      "Number" -> headerNumberToString(line),
      "Type" -> headerTypeToString(line))
    if (line.isFixedCount &&
      (line.getCount == 1 ||
        (line.getType == VCFHeaderLineType.Flag && line.getCount == 0)))
      Field(line.getID, baseType, i, attrs)
    else
      Field(line.getID, TArray(baseType), i, attrs)
  }

  def headerSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T]): Option[TStruct] = {
    if (lines.size > 0)
      Some(TStruct(lines
        .zipWithIndex
        .map { case (line, i) => headerField(line, i) }
        .toArray))
    else None
  }

  def apply[T](hc: HailContext,
    reader: HtsjdkRecordReader[T],
    file1: String,
    files: Array[String] = null,
    nPartitions: Option[Int] = None,
    skipGenotypes: Boolean = false,
    settings: VCFSettings = null,
    genericGenotypes: Boolean = false): LoadVCFResult[T] = {
    val hConf = hc.hadoopConf
    val sc = hc.sc
    val headerLines = hConf.readFile(file1) { s =>
      Source.fromInputStream(s)
        .getLines()
        .takeWhile { line => line(0) == '#' }
        .toArray
    }

    val codec = new htsjdk.variant.vcf.VCFCodec()

    val header = try {
      codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
        .getHeaderValue
        .asInstanceOf[htsjdk.variant.vcf.VCFHeader]
    } catch {
      case e: TribbleException => fatal(
        s"""encountered problem with file $file1
            |  ${ e.getLocalizedMessage }""".stripMargin)
    }

    // FIXME apply descriptions when HTSJDK is fixed to expose filter descriptions
    val filters: Map[String, String] = header
      .getFilterLines
      .toList
      // (filter, description)
      .map(line => (line.getID, ""))
      .toMap

    val infoHeader = header.getInfoHeaderLines
    val infoSignature = headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val genotypeSignature: Type = if (genericGenotypes) headerSignature(formatHeader).getOrElse(TStruct.empty) else TGenotype

    val variantAnnotationSignatures = TStruct(
      Array(
        Some(Field("rsid", TString, 0)),
        Some(Field("qual", TDouble, 1)),
        Some(Field("filters", TSet(TString), 2, filters)),
        Some(Field("pass", TBoolean, 3)),
        infoSignature.map(sig => Field("info", sig, 4))
      ).flatten)

    val headerLine = headerLines.last
    if (!(headerLine(0) == '#' && headerLine(1) != '#'))
      fatal(
        s"""corrupt VCF: expected final header line of format `#CHROM\tPOS\tID...'
            |  found: @1""".stripMargin, headerLine)

    val sampleIds: Array[String] =
      if (skipGenotypes)
        Array.empty
      else
        headerLine
          .split("\t")
          .drop(9)

    val infoSignatureBc = infoSignature.map(sig => sc.broadcast(sig))
    val genotypeSignatureBc = sc.broadcast(genotypeSignature)

    val headerLinesBc = sc.broadcast(headerLines)

    val files2 = if (files == null)
      Array(file1)
    else
      files

    val lines = sc.textFilesLines(files2, nPartitions.getOrElse(sc.defaultMinPartitions))
    val partitionFile = lines.partitions.map(partitionPath)

    val justVariants = lines
      .filter(_.map { line => !line.isEmpty &&
        line(0) != '#' &&
        lineRef(line).forall(c => c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N')
        // FIXME this doesn't filter symbolic, but also avoids decoding the line.  Won't cause errors but might cause unnecessary shuffles
      }.value)
      .map(_.map(lineVariant).value)
    justVariants.persist(StorageLevel.MEMORY_AND_DISK)

    val reportAccs = partitionFile.toSet.iterator
      .map { (file: String) =>
        val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
        VCFReport.accumulators ::= (file, reportAcc)
        (file, reportAcc)
      }
      .toMap

    val rdd = lines
      .mapPartitionsWithIndex { case (i, lines) =>
        val file = partitionFile(i)
        val reportAcc = reportAccs(file)

        val codec = new htsjdk.variant.vcf.VCFCodec()
        codec.readHeader(new BufferedLineIterator(headerLinesBc.value.iterator.buffered))

        lines.flatMap { l => l.map { line =>
          if (line.isEmpty || line(0) == '#')
            None
          else if (!lineRef(line).forall(c => c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N')) {
            reportAcc += VCFReport.RefNonACGTN
            None
          } else {
            val vc = codec.decode(line)
            if (vc.isSymbolic) {
              reportAcc += VCFReport.Symbolic
              None
            } else
              Some(reader.readRecord(codec, reportAcc, vc, infoSignatureBc.map(_.value), genotypeSignatureBc.value, settings))
          }
        }.value
        }
      }.toOrderedRDD(justVariants)

    justVariants.unpersist()

    LoadVCFResult(rdd = rdd, sampleIds = sampleIds, vaSignature = variantAnnotationSignatures, gSignature = genotypeSignature)
  }

  def apply(hc: HailContext,
    file1: String,
    files: Array[String], // FIXME hack
    storeGQ: Boolean,
    nPartitions: Option[Int],
    skipGenotypes: Boolean,
    ppAsPL: Boolean,
    skipBadAD: Boolean): VariantDataset = {

    val settings = VCFSettings(storeGQ, skipGenotypes, ppAsPL, skipBadAD)
    val reader = new GenotypeRecordReader()

    val result = LoadVCF(hc, reader, file1, files, nPartitions, skipGenotypes, settings, genericGenotypes = false)

    VariantSampleMatrix(hc, VariantMetadata(result.sampleIds,
      Annotation.emptyIndexedSeq(result.sampleIds.length),
      Annotation.empty,
      TStruct.empty,
      result.vaSignature,
      TStruct.empty,
      TGenotype,
      isGenericGenotype = false), result.rdd)
  }

  def apply(hc: HailContext,
    file1: String,
    files: Array[String], // FIXME hack
    nPartitions: Option[Int],
    skipGenotypes: Boolean): GenericDataset = {

    val reader = new GenericRecordReader()
    val result = LoadVCF(hc, reader, file1, files, nPartitions, skipGenotypes, settings = null, genericGenotypes = true)

    VariantSampleMatrix(hc, VariantMetadata(result.sampleIds,
      Annotation.emptyIndexedSeq(result.sampleIds.length),
      Annotation.empty,
      TStruct.empty,
      result.vaSignature,
      TStruct.empty,
      result.gSignature,
      isGenericGenotype = true), result.rdd)
  }
}

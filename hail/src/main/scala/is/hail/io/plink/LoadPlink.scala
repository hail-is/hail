package is.hail.io.plink

<<<<<<< HEAD
import is.hail.HailContext
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{ExecuteContext, LowerMatrixIR, MatrixHybridReader, PruneDeadFields, TableRead, TableValue}
=======
import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.ir.{ExecuteContext, GenericTableValue, LowerMatrixIR, MatrixHybridReader, TableRead, TableValue}
>>>>>>> add GenericTableValue, use for plink import
import is.hail.expr.types._
import is.hail.expr.types.physical.{PBoolean, PCanonicalString, PCanonicalStruct, PFloat64, PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.io.vcf.LoadVCF
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import is.hail.io.fs.{FS, Seekable}
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JValue}

case class FamFileConfig(isQuantPheno: Boolean = false,
  delimiter: String = "\\t",
  missingValue: String = "NA")

object LoadPlink {
  def expectedBedSize(nSamples: Int, nVariants: Long): Long = 3 + nVariants * ((nSamples + 3) / 4)

  def parseBim(fs: FS, bimPath: String, a2Reference: Boolean = true,
    contigRecoding: Map[String, String] = Map.empty[String, String]): Array[PlinkVariant] = {
    fs.readLines(bimPath)(_.map(_.map { line =>
      line.split("\\s+") match {
        case Array(contig, rsId, cmPos, bpPos, allele1, allele2) =>
          val recodedContig = contigRecoding.getOrElse(contig, contig)
          if (a2Reference)
            new PlinkVariant(recodedContig, bpPos.toInt, cmPos.toDouble, allele2, allele1, rsId)
          else
            new PlinkVariant(recodedContig, bpPos.toInt, cmPos.toDouble, allele1, allele2, rsId)

        case other => fatal(s"Invalid .bim line.  Expected 6 fields, found ${ other.length } ${ plural(other.length, "field") }")
      }
    }.value
    ).toArray)
  }

  val numericRegex =
    """^-?(?:\d+|\d*\.\d+)(?:[eE]-?\d+)?$""".r

  def parseFam(fs: FS, filename: String, ffConfig: FamFileConfig): (IndexedSeq[Row], PCanonicalStruct) = {

    val delimiter = unescapeString(ffConfig.delimiter)

    val phenoSig = if (ffConfig.isQuantPheno) ("quant_pheno", PFloat64()) else ("is_case", PBoolean())

    val signature = PCanonicalStruct(("id", PCanonicalString()), ("fam_id", PCanonicalString()), ("pat_id", PCanonicalString()),
      ("mat_id", PCanonicalString()), ("is_female", PBoolean()), phenoSig)

    val idBuilder = new ArrayBuilder[String]
    val structBuilder = new ArrayBuilder[Row]

    val m = fs.readLines(filename) {
      _.foreachLine { line =>
        val split = line.split(delimiter)
        if (split.length != 6)
          fatal(s"expected 6 fields, but found ${ split.length }")
        val Array(fam, kid, dad, mom, isFemale, pheno) = split

        val fam1 = if (fam != "0") fam else null
        val dad1 = if (dad != "0") dad else null
        val mom1 = if (mom != "0") mom else null

        val isFemale1 = isFemale match {
          case ffConfig.missingValue => null
          case "-9" => null
          case "0" => null
          case "1" => false
          case "2" => true
          case _ => fatal(s"Invalid sex: '$isFemale'. Male is '1', female is '2', unknown is '0'")
        }

        var warnedAbout9 = false
        val pheno1 =
          if (ffConfig.isQuantPheno)
            pheno match {
              case ffConfig.missingValue => null
              case "-9" =>
                if (!warnedAbout9) {
                  warn(
                    s"""Interpreting value '-9' as a valid quantitative phenotype, which differs from default PLINK behavior.
                       |  Use missing='-9' to interpret '-9' as a missing value.""".stripMargin)
                  warnedAbout9 = true
                }
                -9d
              case numericRegex() => pheno.toDouble
              case _ => fatal(s"Invalid quantitative phenotype: '$pheno'. Value must be numeric or '${ ffConfig.missingValue }'")
            }
          else
            pheno match {
              case ffConfig.missingValue => null
              case "1" => false
              case "2" => true
              case "0" => null
              case "-9" => null
              case "N/A" => null
              case numericRegex() => fatal(s"Invalid case-control phenotype: '$pheno'. Control is '1', case is '2', missing is '0', '-9', '${ ffConfig.missingValue }', or non-numeric.")
              case _ => null
            }
        idBuilder += kid
        structBuilder += Row(kid, fam1, dad1, mom1, isFemale1, pheno1)
      }
    }

    val sampleIds = idBuilder.result()
    LoadVCF.warnDuplicates(sampleIds)

    if (sampleIds.isEmpty)
      fatal("Empty FAM file")

    (structBuilder.result(), signature)
  }
}

object MatrixPLINKReader {
  def fromJValue(ctx: ExecuteContext, jv: JValue): MatrixPLINKReader = {
    val backend = ctx.backend
    val fs = ctx.fs

    implicit val formats: Formats = DefaultFormats
    val params = jv.extract[MatrixPLINKReaderParameters]

    val referenceGenome = params.rg.map(ReferenceGenome.getReference)
    referenceGenome.foreach(_.validateContigRemap(params.contigRecoding))

    val ffConfig = FamFileConfig(params.quantPheno, params.delimiter, params.missing)

    val (sampleInfo, signature) = LoadPlink.parseFam(fs, params.fam, ffConfig)

    val nameMap = Map("id" -> "s")
    val saSignature = signature.copy(fields = signature.fields.map(f => f.copy(name = nameMap.getOrElse(f.name, f.name))))

    val nSamples = sampleInfo.length
    if (nSamples <= 0)
      fatal("FAM file does not contain any samples")

    val variants = LoadPlink.parseBim(fs, params.bim, params.a2Reference, params.contigRecoding)
    val nVariants = variants.length
    if (nVariants <= 0)
      fatal("BIM file does not contain any variants")

    info(s"Found $nSamples samples in fam file.")
    info(s"Found $nVariants variants in bim file.")

    using(fs.open(params.bed)) { dis =>
      val b1 = dis.read()
      val b2 = dis.read()
      val b3 = dis.read()

      if (b1 != 108 || b2 != 27)
        fatal("First two bytes of BED file do not match PLINK magic numbers 108 & 27")

      if (b3 == 0)
        fatal("BED file is in individual major mode. First use plink with --make-bed to convert file to snp major mode before using Hail")
    }

    val bedSize = fs.getFileSize(params.bed)
    if (bedSize != LoadPlink.expectedBedSize(nSamples, nVariants))
      fatal("BED file size does not match expected number of bytes based on BIM and FAM files")

    var nPartitions = params.nPartitions match {
      case Some(nPartitions) => nPartitions
      case None =>
        val blockSizeInB = params.blockSizeInMB.getOrElse(128) * 1024 * 1024
        math.min(nVariants, (nVariants + blockSizeInB - 1) / blockSizeInB)
    }
    params.minPartitions match {
      case Some(minPartitions) =>
        if (nPartitions < minPartitions)
          nPartitions = minPartitions
      case None =>
    }

    val partSize = partition(nVariants, nPartitions)
    val partScan = partSize.scanLeft(0)(_ + _)

    val contexts = Array.tabulate[Any](nPartitions) { i =>
      Row(params.bed, partScan(i), partScan(i + 1))
    }

    val fullMatrixType: MatrixType = MatrixType(
      globalType = TStruct.empty,
      colKey = Array("s"),
      colType = saSignature.virtualType,
      rowType = TStruct(
        "locus" -> TLocus.schemaFromRG(referenceGenome),
        "alleles" -> TArray(TString),
        "rsid" -> TString,
        "cm_position" -> TFloat64),
      rowKey = Array("locus", "alleles"),
      entryType = TStruct("GT" -> TCall))

    new MatrixPLINKReader(params, referenceGenome, fullMatrixType, sampleInfo, variants, contexts)
  }
}

case class MatrixPLINKReaderParameters(
  bed: String,
  bim: String,
  fam: String,
  nPartitions: Option[Int],
  blockSizeInMB: Option[Int],
  minPartitions: Option[Int],
  delimiter: String,
  missing: String,
  quantPheno: Boolean,
  a2Reference: Boolean,
  rg: Option[String],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean)

class PlinkVariant(
  val contig: String,
  val pos: Int,
  val cmPos: Double,
  val ref: String,
  val alt: String,
  val rsid: String)

class MatrixPLINKReader(
  val params: MatrixPLINKReaderParameters,
  referenceGenome: Option[ReferenceGenome],
  val fullMatrixType: MatrixType,
  sampleInfo: IndexedSeq[Row],
  variants: Array[PlinkVariant],
  contexts: Array[Any]
) extends MatrixHybridReader {
  def pathsUsed: Seq[String] = FastSeq(params.bed, params.bim, params.fam)

  def nSamples: Int = sampleInfo.length

  def nVariants: Long = variants.length

  val columnCount: Option[Int] = Some(nSamples)

  val partitionCounts: Option[IndexedSeq[Long]] = None

  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val fsBc = ctx.fsBc

    val localA2Reference = params.a2Reference
    val localSkipInvalidLoci = params.skipInvalidLoci
    val localRG = referenceGenome
    val variantsBc = ctx.backend.broadcast(variants)
    val localNSamples = nSamples

    val globals = Row(sampleInfo)

    val contextType = TStruct(
      "bed" -> TString,
      "start" -> TInt64,
      "end" -> TInt64)

    /* RVD can't handle non-canonical row PTypes yet

    val fullRowPType = PCanonicalStruct(true,
      "locus" -> PCanonicalLocus.schemaFromRG(localRG, true),
      "alleles" -> PCanonicalArray(PCanonicalString(true), true),
      "rsid" -> PCanonicalString(true),
      "cm_position" -> PFloat64(true),
      LowerMatrixIR.entriesFieldName -> PCanonicalArray(PCanonicalStruct(true, "GT" -> PCanonicalCall()), true))

    val bodyPType = (requestedRowType: TStruct) => fullRowPType.subsetTo(requestedRowType).asInstanceOf[PStruct]
     */

    val bodyPType = (requestedRowType: TStruct) => PType.canonical(requestedRowType).setRequired(true).asInstanceOf[PStruct]

    val body = { (requestedType: TStruct) =>
      val hasLocus = requestedType.hasField("locus")
      val hasAlleles = requestedType.hasField("alleles")
      val hasRsid = requestedType.hasField("rsid")
      val hasCmPos = requestedType.hasField("cm_position")

      val hasEntries = requestedType.hasField(LowerMatrixIR.entriesFieldName)
      val hasGT = hasEntries && (requestedType.fieldType(LowerMatrixIR.entriesFieldName).asInstanceOf[TArray]
        .elementType.asInstanceOf[TStruct].hasField("GT"))

      val requestedPType = bodyPType(requestedType)

      { (region: Region, context: Any) =>
        val c = context.asInstanceOf[Row]
        val bed = c.getString(0)
        val start = c.getInt(1)
        val end = c.getInt(2)

        val blockLength = (localNSamples + 3) / 4

        val rvb = new RegionValueBuilder(region)

        val is = fsBc.value.open(bed)
        TaskContext.get.addTaskCompletionListener { (context: TaskContext) =>
          is.close()
        }

        val offset = 3 + start * blockLength
        is match {
          case base: Seekable =>
            base.seek(offset)
          case base: org.apache.hadoop.fs.Seekable =>
            base.seek(offset)
        }

        val input = new Array[Byte](blockLength)

        val table = new Array[Int](4)
        table(0) = if (localA2Reference) Call2.fromUnphasedDiploidGtIndex(2) else Call2.fromUnphasedDiploidGtIndex(0)
        // 1 missing
        table(2) = Call2.fromUnphasedDiploidGtIndex(1)
        table(3) = if (localA2Reference) Call2.fromUnphasedDiploidGtIndex(0) else Call2.fromUnphasedDiploidGtIndex(2)

        Iterator.range(start, end).flatMap { i =>
          val variant = variantsBc.value(i)
          val contig = variant.contig
          val pos = variant.pos

          is.readFully(input, 0, input.length)

          if (localSkipInvalidLoci && !localRG.forall(_.isValidLocus(contig, pos)))
            None
          else {
            rvb.start(requestedPType)
            rvb.startStruct()

            if (hasLocus) {
              // addLocus
              localRG.foreach(_.checkLocus(contig, pos))
              rvb.startStruct()
              rvb.addString(contig)
              rvb.addInt(pos)
              rvb.endStruct()
            }

            if (hasAlleles) {
              rvb.startArray(2)
              rvb.addString(variant.ref)
              rvb.addString(variant.alt)
              rvb.endArray()
            }

            if (hasRsid)
              rvb.addString(variant.rsid)
            if (hasCmPos)
              rvb.addDouble(variant.cmPos)

            if (hasEntries) {
              rvb.startArray(localNSamples)
              if (hasGT) {
                var i = 0
                while (i < localNSamples) {
                  rvb.startStruct() // g
                  val x = (input(i >> 2) >> ((i & 3) << 1)) & 3
                  if (x == 1)
                    rvb.setMissing()
                  else
                    rvb.addInt(table(x))
                  rvb.endStruct() // g
                  i += 1
                }
              } else {
                var i = 0
                while (i < localNSamples) {
                  rvb.startStruct() // g
                  rvb.endStruct() // g
                  i += 1
                }
              }

              rvb.endArray()
            }
            rvb.endStruct()

            Some(rvb.end())
          }
        }
      }
    }

    val tt = fullMatrixType.toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)

    new GenericTableValue(
      tt,
      { (requestedGlobalsType: Type) =>
        val subset = tt.globalType.valueSubsetter(requestedGlobalsType)
        subset(globals).asInstanceOf[Row]
      },
      contextType,
      contexts,
      bodyPType,
      body)
  }

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue =
    executeGeneric(ctx).toTableValue(ctx, tr.typ)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixPLINKReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixPLINKReader => params == that.params
    case _ => false
  }
}

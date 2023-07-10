package is.hail.io.plink

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.asm4s.HailClassLoader
import is.hail.backend.ExecuteContext
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.TableStage
import is.hail.io.fs.{FS, Seekable}
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.RVDPartitioner
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats, JValue}

case class FamFileConfig(isQuantPheno: Boolean = false,
  delimiter: String = "\\t",
  missingValue: String = "NA")

object LoadPlink {
  def expectedBedSize(nSamples: Int, nVariants: Long): Long = 3 + nVariants * ((nSamples + 3) / 4)

  def parseBim(ctx: ExecuteContext, fs: FS, bimPath: String, a2Reference: Boolean,
    contigRecoding: Map[String, String], rg: Option[ReferenceGenome], locusAllelesType: TStruct,
    skipInvalidLoci: Boolean): (Int, Array[PlinkVariant]) = {
    val vs = new BoxedArrayBuilder[PlinkVariant]()
    var n = 0
    fs.readLines(bimPath) { lines =>
      lines.foreach { cline =>
        cline.foreach { line =>
          line.split("\\s+") match {
            case Array(contig, rsId, cmPos, bpPos, allele1, allele2) =>
              val pos = bpPos.toInt
              val recodedContig = contigRecoding.getOrElse(contig, contig)
              if (!skipInvalidLoci || rg.forall(_.isValidLocus(recodedContig, pos))) {
                val locus = Locus.annotation(recodedContig, bpPos.toInt, rg)
                val alleles =
                  if (a2Reference)
                    FastIndexedSeq(allele2, allele1)
                  else
                    FastIndexedSeq(allele1, allele2)
                val locusAlleles = Row(locus, alleles)
                vs += new PlinkVariant(n, locusAlleles, cmPos.toDouble, rsId)
              }

            case _ =>
              fatal(s"Invalid .bim line.  Expected 6 fields, found ${ line.length } ${ plural(line.length, "field") }")
          }
        }
        n += 1
      }
    }
    val variants = vs.result()
    (n, variants.sortBy(_.locusAlleles)(locusAllelesType.ordering(ctx.stateManager).toOrdering))
  }

  val numericRegex =
    """^-?(?:\d+|\d*\.\d+)(?:[eE]-?\d+)?$""".r

  def importFamJSON(fs: FS, path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): String = {
    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)
    val (data, ptyp) = LoadPlink.parseFam(fs, path, ffConfig)
    val jv = JSONAnnotationImpex.exportAnnotation(
      Row(ptyp.virtualType.toString, data),
      TStruct("type" -> TString, "data" -> TArray(ptyp.virtualType)))
    JsonMethods.compact(jv)
  }


  def parseFam(fs: FS, filename: String, ffConfig: FamFileConfig): (IndexedSeq[Row], PCanonicalStruct) = {

    val delimiter = unescapeString(ffConfig.delimiter)

    val phenoSig = if (ffConfig.isQuantPheno) ("quant_pheno", PFloat64()) else ("is_case", PBoolean())

    val signature = PCanonicalStruct(("id", PCanonicalString()), ("fam_id", PCanonicalString()), ("pat_id", PCanonicalString()),
      ("mat_id", PCanonicalString()), ("is_female", PBoolean()), phenoSig)

    val idBuilder = new BoxedArrayBuilder[String]
    val structBuilder = new BoxedArrayBuilder[Row]

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

    val referenceGenome = params.rg.map(ctx.getReference)
    referenceGenome.foreach(_.validateContigRemap(params.contigRecoding))

    val locusType = TLocus.schemaFromRG(params.rg)
    val locusAllelesType = TStruct(
      "locus" -> locusType,
      "alleles" -> TArray(TString))

    val ffConfig = FamFileConfig(params.quantPheno, params.delimiter, params.missing)

    val (sampleInfo, signature) = LoadPlink.parseFam(fs, params.fam, ffConfig)

    val nameMap = Map("id" -> "s")
    val saSignature = signature.copy(fields = signature.fields.map(f => f.copy(name = nameMap.getOrElse(f.name, f.name))))

    val nSamples = sampleInfo.length
    if (nSamples <= 0)
      fatal("FAM file does not contain any samples")

    val (nTotalVariants, variants) = LoadPlink.parseBim(ctx, fs, params.bim, params.a2Reference, params.contigRecoding,
      referenceGenome, locusAllelesType, params.skipInvalidLoci)
    val nVariants = variants.length
    if (nTotalVariants <= 0)
      fatal("BIM file does not contain any variants")

    info(s"Found $nSamples samples in fam file.")
    info(s"Found $nTotalVariants variants in bim file.")

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
    if (bedSize != LoadPlink.expectedBedSize(nSamples, nTotalVariants))
      fatal("BED file size does not match expected number of bytes based on BIM and FAM files")

    var nPartitions = params.nPartitions match {
      case Some(nPartitions) => nPartitions
      case None =>
        val blockSizeInB = params.blockSizeInMB.getOrElse(16) * 1024 * 1024
        ((bedSize + blockSizeInB - 1) / blockSizeInB).toInt
    }
    params.minPartitions match {
      case Some(minPartitions) =>
        if (nPartitions < minPartitions)
          nPartitions = minPartitions
      case None =>
    }
    // partitions non-empty
    if (nPartitions > nVariants)
      nPartitions = nVariants

    val partSize = partition(nVariants, nPartitions)
    val partScan = partSize.scanLeft(0)(_ + _)

    val cb = new BoxedArrayBuilder[Row]()
    val ib = new BoxedArrayBuilder[Interval]()

    var p = 0
    var prevEnd = 0
    val lOrd = locusType.ordering(ctx.stateManager)
    while (p < nPartitions && prevEnd < nVariants) {
      val start = prevEnd

      var end = partScan(p + 1)
      if (start < end) {
        while (end < nVariants
          && lOrd.equiv(variants(end - 1).locusAlleles.asInstanceOf[Row].get(0),
            variants(end).locusAlleles.asInstanceOf[Row].get(0)))
          end += 1

        cb += Row(params.bed, start, end)

        ib += Interval(
          variants(start).locusAlleles,
          variants(end - 1).locusAlleles,
          includesStart = true, includesEnd = true)

        prevEnd = end
      }

      p += 1
    }
    assert(prevEnd == nVariants)

    val contexts = cb.result().map(r => r: Any)

    val partitioner = new RVDPartitioner(ctx.stateManager, locusAllelesType, ib.result(), 0)

    val fullMatrixType: MatrixType = MatrixType(
      globalType = TStruct.empty,
      colKey = Array("s"),
      colType = saSignature.virtualType,
      rowType = TStruct(
        "locus" -> locusType,
        "alleles" -> TArray(TString),
        "rsid" -> TString,
        "cm_position" -> TFloat64),
      rowKey = Array("locus", "alleles"),
      entryType = TStruct("GT" -> TCall))
    assert(locusAllelesType == fullMatrixType.rowKeyStruct)

    new MatrixPLINKReader(params, referenceGenome, fullMatrixType, sampleInfo, variants, contexts, partitioner)
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
  val index: Int,
  val locusAlleles: Any,
  val cmPos: Double,
  val rsid: String
) extends Serializable

class MatrixPLINKReader(
  val params: MatrixPLINKReaderParameters,
  referenceGenome: Option[ReferenceGenome],
  val fullMatrixTypeWithoutUIDs: MatrixType,
  sampleInfo: IndexedSeq[Row],
  variants: Array[PlinkVariant],
  contexts: Array[Any],
  partitioner: RVDPartitioner
) extends MatrixHybridReader {

  def rowUIDType = TInt64
  def colUIDType = TInt64

  def pathsUsed: Seq[String] = FastSeq(params.bed, params.bim, params.fam)

  def nSamples: Int = sampleInfo.length

  def nVariants: Long = variants.length

  val columnCount: Option[Int] = Some(nSamples)

  val partitionCounts: Option[IndexedSeq[Long]] = None

  val globals = Row(sampleInfo.zipWithIndex.map { case (s, idx) =>
    Row((0 until s.length).map(s.apply) :+ idx.toLong :_*)
  })

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PType.canonical(requestedType.rowType).setRequired(true))

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PInt64Required)

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PType.canonical(requestedType.globalType))

  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val localA2Reference = params.a2Reference
    val variantsBc = ctx.backend.broadcast(variants)
    val localNSamples = nSamples
    val sm = ctx.stateManager

    val localLocusType = TLocus.schemaFromRG(referenceGenome.map(_.name))

    val contextType = TStruct(
      "bed" -> TString,
      "start" -> TInt32,
      "end" -> TInt32,
      "partitionIndex" -> TInt32)

    val contextsWithPartIdx = contexts.zipWithIndex.map { case (row: Row, partIdx: Int) =>
      Row(row(0), row(1), row(2), partIdx)
    }

    val fullRowPType = PCanonicalStruct(true,
      "locus" -> PCanonicalLocus.schemaFromRG(referenceGenome.map(_.name), true),
      "alleles" -> PCanonicalArray(PCanonicalString(true), true),
      "rsid" -> PCanonicalString(true),
      "cm_position" -> PFloat64(true),
      LowerMatrixIR.entriesFieldName -> PCanonicalArray(PCanonicalStruct(true, "GT" -> PCanonicalCall()), true),
      rowUIDFieldName -> PInt64Required)

    val bodyPType = (requestedRowType: TStruct) => fullRowPType.subsetTo(requestedRowType).asInstanceOf[PStruct]

    val body = { (requestedType: TStruct) =>
      val hasLocus = requestedType.hasField("locus")
      val hasAlleles = requestedType.hasField("alleles")
      val hasRsid = requestedType.hasField("rsid")
      val hasCmPos = requestedType.hasField("cm_position")
      val hasRowUID = requestedType.hasField(rowUIDFieldName)

      val hasEntries = requestedType.hasField(LowerMatrixIR.entriesFieldName)
      val hasGT = hasEntries && (requestedType.fieldType(LowerMatrixIR.entriesFieldName).asInstanceOf[TArray]
        .elementType.asInstanceOf[TStruct].hasField("GT"))

      val requestedPType = bodyPType(requestedType)

      { (region: Region, theHailClassLoader: HailClassLoader, fs: FS, context: Any) =>
        val c = context.asInstanceOf[Row]
        val bed = c.getString(0)
        val start = c.getInt(1)
        val end = c.getInt(2)

        val blockLength = (localNSamples + 3) / 4

        val rvb = new RegionValueBuilder(sm, region)

        val is = fs.open(bed)
        if (TaskContext.get != null) {
          // FIXME: need to close InputStream for other backends too
          TaskContext.get.addTaskCompletionListener[Unit] { (context: TaskContext) =>
            is.close()
          }
        }
        var offset: Long = 0

        val input = new Array[Byte](blockLength)

        val table = new Array[Int](4)
        table(0) = if (localA2Reference) Call2.fromUnphasedDiploidGtIndex(2) else Call2.fromUnphasedDiploidGtIndex(0)
        // 1 missing
        table(2) = Call2.fromUnphasedDiploidGtIndex(1)
        table(3) = if (localA2Reference) Call2.fromUnphasedDiploidGtIndex(0) else Call2.fromUnphasedDiploidGtIndex(2)

        Iterator.range(start, end).flatMap { i =>
          val variant = variantsBc.value(i)

          val newOffset: Long = 3L + variant.index.toLong * blockLength
          if (newOffset != offset) {
            is match {
              case base: Seekable =>
                base.seek(newOffset)
              case base: org.apache.hadoop.fs.Seekable =>
                base.seek(newOffset)
            }
            offset = newOffset
          }

          is.readFully(input, 0, input.length)

          rvb.start(requestedPType)
          rvb.startStruct()

          val locusAlleles = variant.locusAlleles.asInstanceOf[Row]

          if (hasLocus)
            rvb.addAnnotation(localLocusType, locusAlleles.get(0))

          if (hasAlleles) {
            val alleles = locusAlleles.getAs[IndexedSeq[String]](1)
            rvb.startArray(2)
            rvb.addString(alleles(0))
            rvb.addString(alleles(1))
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
                  rvb.addCall(table(x))
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

          if (hasRowUID)
            rvb.addLong(i)

          rvb.endStruct()

          Some(rvb.end())
        }
      }
    }

    val tt = matrixToTableType(fullMatrixType)

    new GenericTableValue(
      tt,
      rowUIDFieldName,
      Some(partitioner),
      { (requestedGlobalsType: Type) =>
        val subset = tt.globalType.valueSubsetter(requestedGlobalsType)
        subset(globals).asInstanceOf[Row]
      },
      contextType,
      contextsWithPartIdx,
      bodyPType,
      body)
  }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val tt = fullMatrixType.toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)
    val subset = tt.globalType.valueSubsetter(requestedGlobalsType)
    Literal(requestedGlobalsType, subset(globals).asInstanceOf[Row])
  }

  override def _lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType, "PLINK file", params)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixPLINKReader")
  }

  def renderShort(): String = defaultRender()

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixPLINKReader => params == that.params
    case _ => false
  }
}

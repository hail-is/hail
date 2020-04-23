package is.hail.io.plink

import is.hail.HailContext
import is.hail.expr.ir.{ExecuteContext, LowerMatrixIR, MatrixHybridReader, PruneDeadFields, TableRead, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.physical.{PBoolean, PCanonicalString, PCanonicalStruct, PFloat64}
import is.hail.expr.types.virtual._
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.RVD
import is.hail.sparkextras.ContextRDD
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant.{Locus, _}
import is.hail.io.fs.FS
import org.apache.hadoop.io.LongWritable
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Extraction, Formats, JObject, JValue}

case class FamFileConfig(isQuantPheno: Boolean = false,
  delimiter: String = "\\t",
  missingValue: String = "NA")

object LoadPlink {
  def expectedBedSize(nSamples: Int, nVariants: Long): Long = 3 + nVariants * ((nSamples + 3) / 4)

  def parseBim(fs: FS, bimPath: String, a2Reference: Boolean = true,
    contigRecoding: Map[String, String] = Map.empty[String, String]): Array[(String, Int, Double, String, String, String)] = {
    fs.readLines(bimPath)(_.map(_.map { line =>
      line.split("\\s+") match {
        case Array(contig, rsId, cmPos, bpPos, allele1, allele2) =>
          val recodedContig = contigRecoding.getOrElse(contig, contig)
          if (a2Reference)
            (recodedContig, bpPos.toInt, cmPos.toDouble, allele2, allele1, rsId)
          else
            (recodedContig, bpPos.toInt, cmPos.toDouble, allele1, allele2, rsId)

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

    val requestedPartitions = params.nPartitions.getOrElse(backend.defaultParallelism)
    if (bedSize < requestedPartitions)
      fatal(s"The number of partitions requested ($requestedPartitions) is greater than the file size ($bedSize)")

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

    new MatrixPLINKReader(params, referenceGenome, fullMatrixType, sampleInfo, variants)
  }
}

case class MatrixPLINKReaderParameters(
  bed: String,
  bim: String,
  fam: String,
  nPartitions: Option[Int] = None,
  delimiter: String,
  missing: String,
  quantPheno: Boolean,
  a2Reference: Boolean,
  rg: Option[String],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean)

class MatrixPLINKReader(
  val params: MatrixPLINKReaderParameters,
  referenceGenome: Option[ReferenceGenome],
  val fullMatrixType: MatrixType,
  sampleInfo: IndexedSeq[Row],
  variants: Array[(String, Int, Double, String, String, String)]
) extends MatrixHybridReader {
  def pathsUsed: Seq[String] = FastSeq(params.bed, params.bim, params.fam)

  def nSamples: Int = sampleInfo.length

  def nVariants: Long = variants.length

  val columnCount: Option[Int] = Some(nSamples)

  val partitionCounts: Option[IndexedSeq[Long]] = None

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val backend = ctx.backend
    val sc = HailContext.sc

    val requestedType = tr.typ
    assert(PruneDeadFields.isSupertype(requestedType, fullType))

    val rvd = if (tr.dropRows)
      RVD.empty(sc, requestedType.canonicalRVDType)
    else {
      val variantsBc = ctx.backend.broadcast(variants)
      sc.hadoopConfiguration.setInt("nSamples", nSamples)
      sc.hadoopConfiguration.setBoolean("a2Reference", params.a2Reference)

      val crdd = ContextRDD.weaken(
        sc.hadoopFile(
          params.bed,
          classOf[PlinkInputFormat],
          classOf[LongWritable],
          classOf[PlinkRecord],
          params.nPartitions.getOrElse(backend.defaultParallelism)))

      val kType = requestedType.canonicalRVDType.kType
      val rvRowType = requestedType.canonicalRowPType

      val hasRsid = requestedType.rowType.hasField("rsid")
      val hasCmPos = requestedType.rowType.hasField("cm_position")

      val (hasGT, dropSamples) = requestedType.rowType.fieldOption(LowerMatrixIR.entriesFieldName) match {
        case Some(fd) => fd.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct].hasField("GT") -> false
        case None => false -> true
      }


      val skipInvalidLociLocal = params.skipInvalidLoci
      val rgLocal = referenceGenome

      val fastKeys = crdd.cmapPartitions { (ctx, it) =>
        val rvb = ctx.rvb

        it.flatMap { case (_, record) =>
          val (contig, pos, _, ref, alt, _) = variantsBc.value(record.getKey)
          if (skipInvalidLociLocal && !rgLocal.forall(_.isValidLocus(contig, pos)))
            None
          else {
            rvb.start(kType)
            rvb.startStruct()
            rvb.addAnnotation(kType.types(0).virtualType, Locus.annotation(contig, pos, rgLocal))
            rvb.startArray(2)
            rvb.addString(ref)
            rvb.addString(alt)
            rvb.endArray()
            rvb.endStruct()

            Some(rvb.end())
          }
        }
      }

      val rdd2 = crdd.cmapPartitions { (ctx, it) =>
        val rvb = ctx.rvb

        it.flatMap { case (_, record) =>
          val (contig, pos, cmPos, ref, alt, rsid) = variantsBc.value(record.getKey)

          if (skipInvalidLociLocal && !rgLocal.forall(_.isValidLocus(contig, pos)))
            None
          else {
            rvb.start(rvRowType)
            rvb.startStruct()
            rvb.addAnnotation(kType.types(0).virtualType, Locus.annotation(contig, pos, rgLocal))
            rvb.startArray(2)
            rvb.addString(ref)
            rvb.addString(alt)
            rvb.endArray()
            if (hasRsid)
              rvb.addAnnotation(rvRowType.types(2).virtualType, rsid)
            if (hasCmPos)
              rvb.addDouble(cmPos)
            if (!dropSamples)
              record.getValue(rvb, hasGT)
            rvb.endStruct()

            Some(rvb.end())
          }
        }
      }

      RVD.coerce(ctx, requestedType.canonicalRVDType, rdd2, fastKeys)
    }

    if (params.skipInvalidLoci && referenceGenome.isDefined) {
      val nFiltered = rvd.count() - nVariants
      if (nFiltered > 0)
        info(s"Filtered out $nFiltered ${ plural(nFiltered, "variant") } that are inconsistent with reference genome '${ referenceGenome.get.name }'.")
    }


    val globalValue = makeGlobalValue(ctx, requestedType, sampleInfo)

    TableValue(ctx, requestedType, globalValue, rvd)
  }

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

package is.hail.io.plink

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.OrderedRVD
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.spark.sql.Row

case class SampleInfo(sampleIds: Array[String], annotations: IndexedSeq[Annotation], signatures: TStruct)

case class FamFileConfig(isQuantPheno: Boolean = false,
  delimiter: String = "\\t",
  missingValue: String = "NA")

object PlinkLoader {
  def expectedBedSize(nSamples: Int, nVariants: Long): Long = 3 + nVariants * ((nSamples + 3) / 4)

  private def parseBim(bimPath: String, hConf: Configuration, a2Reference: Boolean = true,
    contigRecoding: Map[String, String] = Map.empty[String, String]): Array[(Variant, String)] = {
    hConf.readLines(bimPath)(_.map(_.map { line =>
      line.split("\\s+") match {
        case Array(contig, rsId, morganPos, bpPos, allele1, allele2) =>
          val recodedContig = contigRecoding.getOrElse(contig, contig)

          if (a2Reference)
            (Variant(recodedContig, bpPos.toInt, allele2, allele1), rsId)
          else
            (Variant(recodedContig, bpPos.toInt, allele1, allele2), rsId)

        case other => fatal(s"Invalid .bim line.  Expected 6 fields, found ${ other.length } ${ plural(other.length, "field") }")
      }
    }.value
    ).toArray)
  }

  val numericRegex =
    """^-?(?:\d+|\d*\.\d+)(?:[eE]-?\d+)?$""".r

  def parseFam(filename: String, ffConfig: FamFileConfig,
    hConf: hadoop.conf.Configuration): (IndexedSeq[Row], TStruct) = {

    val delimiter = unescapeString(ffConfig.delimiter)

    val phenoSig = if (ffConfig.isQuantPheno) ("quant_pheno", TFloat64()) else ("is_case", TBoolean())

    val signature = TStruct(("id", TString()), ("fam_id", TString()), ("pat_id", TString()),
      ("mat_id", TString()), ("is_female", TBoolean()), phenoSig)

    val idBuilder = new ArrayBuilder[String]
    val structBuilder = new ArrayBuilder[Row]

    val m = hConf.readLines(filename) {
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
          case _ => fatal(s"Invalid sex: `$isFemale'. Male is `1', female is `2', unknown is `0'")
        }

        val pheno1 =
          if (ffConfig.isQuantPheno)
            pheno match {
              case ffConfig.missingValue => null
              case numericRegex() => pheno.toDouble
              case _ => fatal(s"Invalid quantitative phenotype: `$pheno'. Value must be numeric or `${ ffConfig.missingValue }'")
            }
          else
            pheno match {
              case ffConfig.missingValue => null
              case "1" => false
              case "2" => true
              case "0" => null
              case "-9" => null
              case "N/A" => null
              case numericRegex() => fatal(s"Invalid case-control phenotype: `$pheno'. Control is `1', case is `2', missing is `0', `-9', `${ ffConfig.missingValue }', or non-numeric.")
              case _ => null
            }
        idBuilder += kid
        structBuilder += Row(kid, fam1, dad1, mom1, isFemale1, pheno1)
      }
    }

    val sampleIds = idBuilder.result()
    LoadVCF.warnDuplicates(sampleIds)

    if (sampleIds.isEmpty)
      fatal("Empty .fam file")

    (structBuilder.result(), signature)
  }

  private def parseBed(hc: HailContext,
    bedPath: String,
    sampleAnnotations: IndexedSeq[Annotation],
    sampleAnnotationSignature: TStruct,
    variants: Array[(Variant, String)],
    nPartitions: Option[Int] = None,
    a2Reference: Boolean = true,
    gr: GenomeReference = GenomeReference.defaultReference,
    dropChr0: Boolean = false): MatrixTable = {

    val sc = hc.sc
    val nSamples = sampleAnnotations.length
    val variantsBc = sc.broadcast(variants)
    sc.hadoopConfiguration.setInt("nSamples", nSamples)
    sc.hadoopConfiguration.setBoolean("a2Reference", a2Reference)

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[PlinkRecord],
      nPartitions.getOrElse(sc.defaultMinPartitions))

    val matrixType = MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = sampleAnnotationSignature,
      rowType = TStruct("locus" -> TLocus(gr), "alleles" -> TArray(TStringRequired), "rsid" -> TString()),
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = TStruct(required = true, "GT" -> TCall()))

    val kType = matrixType.orderedRVType.kType
    val rvRowType = matrixType.rvRowType

    val fastKeys = rdd.mapPartitions { it =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      it.flatMap { case (_, record) =>
        val (v, _) = variantsBc.value(record.getKey)

        if (dropChr0 && v.contig == "0")
          None
        else {
          region.clear()
          rvb.start(kType)
          rvb.startStruct()
          rvb.addAnnotation(kType.fieldType(0), v.locus) // locus/pk
          rvb.startArray(2)
          rvb.addString(v.ref)
          rvb.addString(v.alt)
          rvb.endArray()
          rvb.endStruct()

          rv.setOffset(rvb.end())
          Some(rv)
        }
      }
    }

    val rdd2 = rdd.mapPartitions { it =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      it.flatMap { case (_, record) =>
        val (v, rsid) = variantsBc.value(record.getKey)

        if (dropChr0 && v.contig == "0")
          None
        else {
          region.clear()
          rvb.start(rvRowType)
          rvb.startStruct()
          rvb.addAnnotation(rvRowType.fieldType(0), v.locus) // locus/pk
          rvb.addAnnotation(rvRowType.fieldType(1), IndexedSeq(v.ref, v.alt))
          rvb.addAnnotation(rvRowType.fieldType(2), rsid)
          record.getValue(rvb)
          rvb.endStruct()

          rv.setOffset(rvb.end())
          Some(rv)
        }
      }
    }

    new MatrixTable(hc, matrixType,
      MatrixLocalValue(globalAnnotation = Annotation.empty,
        sampleAnnotations = sampleAnnotations),
      OrderedRVD(matrixType.orderedRVType, rdd2, Some(fastKeys), None))
  }

  def apply(hc: HailContext, bedPath: String, bimPath: String, famPath: String, ffConfig: FamFileConfig,
    nPartitions: Option[Int] = None, a2Reference: Boolean = true, gr: GenomeReference = GenomeReference.defaultReference,
    contigRecoding: Map[String, String] = Map.empty[String, String], dropChr0: Boolean = false): MatrixTable = {
    val (sampleInfo, signature) = parseFam(famPath, ffConfig, hc.hadoopConf)

    val nameMap = Map("id" -> "s")
    val saSignature = signature.copy(fields = signature.fields.map(f => f.copy(name = nameMap.getOrElse(f.name, f.name))))

    val nSamples = sampleInfo.length
    if (nSamples <= 0)
      fatal(".fam file does not contain any samples")

    val variants = parseBim(bimPath, hc.hadoopConf, a2Reference, contigRecoding)
    val nVariants = variants.length
    if (nVariants <= 0)
      fatal(".bim file does not contain any variants")

    info(s"Found $nSamples samples in fam file.")
    info(s"Found $nVariants variants in bim file.")

    hc.sc.hadoopConfiguration.readFile(bedPath) { dis =>
      val b1 = dis.read()
      val b2 = dis.read()
      val b3 = dis.read()

      if (b1 != 108 || b2 != 27)
        fatal("First two bytes of bed file do not match PLINK magic numbers 108 & 27")

      if (b3 == 0)
        fatal("Bed file is in individual major mode. First use plink with --make-bed to convert file to snp major mode before using Hail")
    }

    val bedSize = hc.sc.hadoopConfiguration.getFileSize(bedPath)
    if (bedSize != expectedBedSize(nSamples, nVariants))
      fatal("bed file size does not match expected number of bytes based on bed and fam files")

    if (bedSize < nPartitions.getOrElse(hc.sc.defaultMinPartitions))
      fatal(s"The number of partitions requested (${ nPartitions.getOrElse(hc.sc.defaultMinPartitions) }) is greater than the file size ($bedSize)")

    val vds = parseBed(hc, bedPath, sampleInfo, saSignature, variants, nPartitions, a2Reference, gr, dropChr0)
    vds
  }

}

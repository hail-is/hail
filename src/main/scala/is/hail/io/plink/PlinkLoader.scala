package is.hail.io.plink

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable

import scala.collection.mutable
import scala.reflect.classTag

case class SampleInfo(sampleIds: Array[String], annotations: IndexedSeq[Annotation], signatures: TStruct)

case class FamFileConfig(isQuantitative: Boolean = false,
  delimiter: String = "\\t",
  missingValue: String = "NA")

object PlinkLoader {
  def expectedBedSize(nSamples: Int, nVariants: Long): Long = 3 + nVariants * ((nSamples + 3) / 4)

  val plinkSchema = TStruct(("rsid", TString))

  private def parseBim(bimPath: String, hConf: Configuration): Array[(Variant, String)] = {
    hConf.readLines(bimPath)(_.map(_.map { line =>
      line.split("\\s+") match {
        case Array(contig, rsId, morganPos, bpPos, allele1, allele2) =>
          val recodedContig = contig match {
            case "23" => "X"
            case "24" => "Y"
            case "25" => "X"
            case "26" => "MT"
            case x => x
          }
          (Variant(recodedContig, bpPos.toInt, allele2, allele1), rsId)
        case other => fatal(s"Invalid .bim line.  Expected 6 fields, found ${ other.length } ${ plural(other.length, "field") }")
      }
    }.value
    ).toArray)
  }

  val numericRegex =
    """^-?(?:\d+|\d*\.\d+)(?:[eE]-?\d+)?$""".r

  def parseFam(filename: String, ffConfig: FamFileConfig,
    hConf: hadoop.conf.Configuration): (IndexedSeq[(String, Annotation)], TStruct) = {

    val delimiter = unescapeString(ffConfig.delimiter)

    val phenoSig = if (ffConfig.isQuantitative) ("qPheno", TFloat64) else ("isCase", TBoolean)

    val signature = TStruct(("famID", TString), ("patID", TString), ("matID", TString), ("isFemale", TBoolean), phenoSig)

    val idBuilder = new ArrayBuilder[String]
    val structBuilder = new ArrayBuilder[Annotation]

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
          if (ffConfig.isQuantitative)
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
        structBuilder += Annotation(fam1, dad1, mom1, isFemale1, pheno1)
      }
    }

    val (sampleIds, duplicates) = mangle(idBuilder.result(), "D" * _)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.length } duplicate ${ plural(duplicates.length, "sample ID") }. Mangled IDs follows:\n  @1",
        duplicates.map { case (pre, post) => s"'$pre' -> '$post'"}.truncatable("\n  "))
    }

    if (sampleIds.isEmpty)
      fatal("Empty .fam file")

    (sampleIds.zip(structBuilder.result()), signature)
  }

  private def parseBed(hc: HailContext,
    bedPath: String,
    sampleIds: IndexedSeq[String],
    sampleAnnotations: IndexedSeq[Annotation],
    sampleAnnotationSignature: Type,
    variants: Array[(Variant, String)],
    nPartitions: Option[Int] = None): GenericDataset = {

    val sc = hc.sc
    val nSamples = sampleIds.length
    val variantsBc = sc.broadcast(variants)
    sc.hadoopConfiguration.setInt("nSamples", nSamples)

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[PlinkRecord],
      nPartitions.getOrElse(sc.defaultMinPartitions))

    val fastKeys = rdd.map { case (_, decoder) => variantsBc.value(decoder.getKey)._1: Annotation }
    val variantRDD = rdd.map {
      case (_, vr) =>
        val (v, rsId) = variantsBc.value(vr.getKey)
        (v: Annotation, (Annotation(rsId), vr.getValue: Iterable[Annotation]))
    }.toOrderedRDD(fastKeys)(TVariant.orderedKey, classTag[(Annotation, Iterable[Annotation])])

    new GenericDataset(hc, VSMMetadata(
      saSignature = sampleAnnotationSignature,
      vaSignature = plinkSchema,
      globalSignature = TStruct.empty,
      genotypeSignature = TStruct("GT" -> TCall),
      wasSplit = true),
      VSMLocalValue(globalAnnotation = Annotation.empty,
        sampleIds = sampleIds,
        sampleAnnotations = sampleAnnotations),
      variantRDD)
  }

  def apply(hc: HailContext, bedPath: String, bimPath: String, famPath: String, ffConfig: FamFileConfig,
    nPartitions: Option[Int] = None): GenericDataset = {
    val (sampleInfo, signature) = parseFam(famPath, ffConfig, hc.hadoopConf)
    val nSamples = sampleInfo.length
    if (nSamples <= 0)
      fatal(".fam file does not contain any samples")

    val variants = parseBim(bimPath, hc.hadoopConf)
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

    val (ids, annotations) = sampleInfo.unzip

    val duplicateIds = ids.duplicates().toArray
    if (duplicateIds.nonEmpty) {
      val n = duplicateIds.length
      warn(
        s"""found $n duplicate sample ${ plural(n, "ID") }
           |  Duplicate IDs: @1""".stripMargin, duplicateIds)
    }

    val vds = parseBed(hc, bedPath, ids, annotations, signature, variants, nPartitions)
    vds
  }

}
package is.hail.methods

import java.io.{FileInputStream, IOException}
import java.util.Properties

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.expr.{JSONAnnotationImpex, Parser}
import is.hail.rvd.{OrderedRVD, OrderedRVType}
import is.hail.utils._
import is.hail.variant.{Locus, MatrixTable, RegionValueVariant, Variant}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._


object Nirvana {

  //NOTE THIS SCHEMA IS FOR NIRVANA 1.6.2 as of JUNE 19th
  val nirvanaSignature = TStruct(
    "chromosome" -> TString(),
    "refAllele" -> TString(),
    "position" -> TInt32(),
    "altAlleles" -> TArray(TString()),
    "cytogeneticBand" -> TString(),
    "quality" -> TFloat64(),
    "filters" -> TArray(TString()),
    "jointSomaticNormalQuality" -> TInt32(),
    "copyNumber" -> TInt32(),
    "strandBias" -> TFloat64(),
    "recalibratedQuality" -> TFloat64(),
    "variants" -> TArray(TStruct(
      "altAllele" -> TString(),
      "refAllele" -> TString(),
      "chromosome" -> TString(),
      "begin" -> TInt32(),
      "end" -> TInt32(),
      "phylopScore" -> TFloat64(),
      "isReferenceMinor" -> TBoolean(),
      "variantType" -> TString(),
      "vid" -> TString(),
      "isRecomposed" -> TBoolean(),
      "regulatoryRegions" -> TArray(TStruct(
        "id" -> TString(),
        "consequence" -> TSet(TString()),
        "type" -> TString()
      )),
      "clinvar" -> TArray(TStruct(
        "id" -> TString(),
        "reviewStatus" -> TString(),
        "isAlleleSpecific" -> TBoolean(),
        "alleleOrigins" -> TArray(TString()),
        "refAllele" -> TString(),
        "altAllele" -> TString(),
        "phenotypes" -> TArray(TString()),
        "medGenIds" -> TArray(TString()),
        "omimIds" -> TArray(TString()),
        "orphanetIds" -> TArray(TString()),
        "geneReviewsId" -> TString(),
        "significance" -> TString(),
        "lastUpdatedDate" -> TString(),
        "pubMedIds" -> TArray(TString())
      )),
      "cosmic" -> TArray(TStruct(
        "id" -> TString(),
        "isAlleleSpecific" -> TBoolean(),
        "refAllele" -> TString(),
        "altAllele" -> TString(),
        "gene" -> TString(),
        "sampleCount" -> TInt32(),
        "studies" -> TArray(TStruct(
          "id" -> TInt32(),
          "histology" -> TString(),
          "primarySite" -> TString()
        ))
      )),
      "dbsnp" -> TStruct("ids" -> TArray(TString())),
      "evs" -> TStruct(
        "coverage" -> TInt32(),
        "sampleCount" -> TInt32(),
        "allAf" -> TFloat64(),
        "afrAf" -> TFloat64(),
        "eurAf" -> TFloat64()
      ),
      "exac" -> TStruct(
        "coverage" -> TInt32(),
        "allAf" -> TFloat64(),
        "allAc" -> TInt32(),
        "allAn" -> TInt32(),
        "afrAf" -> TFloat64(),
        "afrAc" -> TInt32(),
        "afrAn" -> TInt32(),
        "amrAf" -> TFloat64(),
        "amrAc" -> TInt32(),
        "amrAn" -> TInt32(),
        "easAf" -> TFloat64(),
        "easAc" -> TInt32(),
        "easAn" -> TInt32(),
        "finAf" -> TFloat64(),
        "finAc" -> TInt32(),
        "finAn" -> TInt32(),
        "nfeAf" -> TFloat64(),
        "nfeAc" -> TInt32(),
        "nfeAn" -> TInt32(),
        "othAf" -> TFloat64(),
        "othAc" -> TInt32(),
        "othAn" -> TInt32(),
        "sasAf" -> TFloat64(),
        "sasAc" -> TInt32(),
        "sasAn" -> TInt32()
      ),
      "globalAllele" -> TStruct(
        "globalMinorAllele" -> TString(),
        "globalMinorAlleleFrequency" -> TFloat64()
      ),
      "oneKg" -> TStruct(
        "ancestralAllele" -> TString(),
        "allAf" -> TFloat64(),
        "allAc" -> TInt32(),
        "allAn" -> TInt32(),
        "afrAf" -> TFloat64(),
        "afrAc" -> TInt32(),
        "afrAn" -> TInt32(),
        "amrAf" -> TFloat64(),
        "amrAc" -> TInt32(),
        "amrAn" -> TInt32(),
        "easAf" -> TFloat64(),
        "easAc" -> TInt32(),
        "easAn" -> TInt32(),
        "eurAf" -> TFloat64(),
        "eurAc" -> TInt32(),
        "eurAn" -> TInt32(),
        "sasAf" -> TFloat64(),
        "sasAc" -> TInt32(),
        "sasAn" -> TInt32()
      ),
      "transcripts" -> TStruct(
        "refSeq" -> TArray(TStruct(
          "transcript" -> TString(),
          "bioType" -> TString(),
          "aminoAcids" -> TString(),
          "cDnaPos" -> TString(),
          "codons" -> TString(),
          "cdsPos" -> TString(),
          "exons" -> TString(),
          "introns" -> TString(),
          "geneId" -> TString(),
          "hgnc" -> TString(),
          "consequence" -> TArray(TString()),
          "hgvsc" -> TString(),
          "hgvsp" -> TString(),
          "isCanonical" -> TBoolean(),
          "polyPhenScore" -> TFloat64(),
          "polyPhenPrediction" -> TString(),
          "proteinId" -> TString(),
          "proteinPos" -> TString(),
          "siftScore" -> TFloat64(),
          "siftPrediction" -> TString()
        )),
        "ensembl" -> TArray(TStruct(
          "transcript" -> TString(),
          "bioType" -> TString(),
          "aminoAcids" -> TString(),
          "cDnaPos" -> TString(),
          "codons" -> TString(),
          "cdsPos" -> TString(),
          "exons" -> TString(),
          "introns" -> TString(),
          "geneId" -> TString(),
          "hgnc" -> TString(),
          "consequence" -> TArray(TString()),
          "hgvsc" -> TString(),
          "hgvsp" -> TString(),
          "isCanonical" -> TBoolean(),
          "polyPhenScore" -> TFloat64(),
          "polyPhenPrediction" -> TString(),
          "proteinId" -> TString(),
          "proteinPos" -> TString(),
          "siftScore" -> TFloat64(),
          "siftPrediction" -> TString()
        ))
      ),
      "genes" -> TArray(TStruct(
        "name" -> TString(),
        "omim" -> TArray(TStruct(
          "mimNumber" -> TInt32(),
          "hgnc" -> TString(),
          "description" -> TString(),
          "phenotypes" -> TArray(TStruct(
            "mimNumber" -> TInt32(),
            "phenotype" -> TString(),
            "mapping" -> TString(),
            "inheritance" -> TArray(TString()),
            "comments" -> TString()
          ))
        ))
      ))
    ))
  )

  def printContext(w: (String) => Unit) {
    w("##fileformat=VCFv4.1")
    w("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
  }

  def printElement(vaSignature: Type)(w: (String) => Unit, v: Variant) {
    val sb = new StringBuilder()
    sb.append(v.contig)
    sb += '\t'
    sb.append(v.start)
    sb.append("\t.\t")
    sb.append(v.ref)
    sb += '\t'
    sb.append(v.altAlleles.iterator.map(_.alt).mkString(","))
    sb += '\t'
    sb.append("\t.\t.\tGT")
    w(sb.result())
  }

  def variantFromInput(contig: String, start: Int, ref: String, altAlleles: Array[String]): Variant = {
    Variant(contig, start, ref, altAlleles)
  }

  def annotate(vds: MatrixTable, config: String, blockSize: Int, root: String = "va.nirvana"): MatrixTable = {
    val parsedRoot = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    val properties = try {
      val p = new Properties()
      val is = new FileInputStream(config)
      p.load(is)
      is.close()
      p
    } catch {
      case e: IOException =>
        fatal(s"could not open file: ${ e.getMessage }")
    }

    val dotnet = properties.getProperty("hail.nirvana.dotnet", "dotnet")

    val nirvanaLocation = properties.getProperty("hail.nirvana.location")
    if (nirvanaLocation == null)
      fatal("property `hail.nirvana.location' required")

    val path = Option(properties.getProperty("hail.nirvana.path"))

    val cache = properties.getProperty("hail.nirvana.cache")


    val supplementaryAnnotationDirectoryOpt = Option(properties.getProperty("hail.nirvana.supplementaryAnnotationDirectory"))
    val supplementaryAnnotationDirectory = if(supplementaryAnnotationDirectoryOpt.isEmpty) List[String]() else List("--sd", supplementaryAnnotationDirectoryOpt.get)

    val reference = properties.getProperty("hail.nirvana.reference")

    val cmd: List[String] = List[String](dotnet, s"$nirvanaLocation") ++
      List("-c", cache) ++
      supplementaryAnnotationDirectory ++
      List("-r", reference,
      "-i", "-",
      "-o", "-")

    println(cmd.mkString(" "))

    val contigQuery: Querier = nirvanaSignature.query("chromosome")
    val startQuery = nirvanaSignature.query("position")
    val refQuery = nirvanaSignature.query("refAllele")
    val altsQuery = nirvanaSignature.query("altAlleles")
    val oldSignature = vds.rowType
    val localBlockSize = blockSize

    implicit val variantOrd = vds.genomeReference.variantOrdering

    info("Running Nirvana")


    val localRowType = vds.rvRowType
    val annotations = vds.rvd
      .mapPartitions { it =>
        val pb = new ProcessBuilder(cmd.asJava)
        val env = pb.environment()
        if (path.orNull != null)
          env.put("PATH", path.get)

        val rvv = new RegionValueVariant(localRowType)

        it.map { rv =>
          rvv.setRegion(rv)
          rvv.variantObject()
        }
          .grouped(localBlockSize)
          .flatMap { block =>
            val (jt, proc) = block.iterator.pipe(pb,
              printContext,
              printElement(oldSignature),
              _ => ())
            // The filter is because every other output line is a comma.
            val kt = jt.filter(_.startsWith("{\"chromosome")).map { s =>
                val a = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(s), nirvanaSignature)
                val v = variantFromInput(contigQuery(a).asInstanceOf[String],
                  startQuery(a).asInstanceOf[Int],
                  refQuery(a).asInstanceOf[String],
                  altsQuery(a).asInstanceOf[Seq[String]].toArray
                )
                (v, a)
              }

            val r = kt.toArray
              .sortBy(_._1)

            val rc = proc.waitFor()
            if (rc != 0)
              fatal(s"nirvana command failed with non-zero exit status $rc")

            r
          }
      }
      .persist(StorageLevel.MEMORY_AND_DISK)


    info(s"nirvana: annotated ${ annotations.count() } variants")

    val nirvanaOrderedRVType = new OrderedRVType(
      Array("locus"), Array("locus", "alleles"),
      TStruct(
        "locus" -> vds.rowKeyTypes(0),
        "alleles" -> vds.rowKeyTypes(1),
        "nirvana" -> nirvanaSignature))

    val nirvanaRowType = nirvanaOrderedRVType.rowType

    val nirvanaRVD: OrderedRVD = OrderedRVD(
      nirvanaOrderedRVType,
      vds.rvd.partitioner,
      annotations.mapPartitions { it =>
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        it.map { case (v, nirvana) =>
          rvb.start(nirvanaRowType)
          rvb.startStruct()
          rvb.startStruct()
          rvb.addAnnotation(nirvanaRowType.fieldType(0), v.locus)
          rvb.addAnnotation(nirvanaRowType.fieldType(1), IndexedSeq(v.ref) ++ v.altAlleles.map(_.alt))
          rvb.addAnnotation(nirvanaRowType.fieldType(2), nirvana)
          rvb.endStruct()
          rv.setOffset(rvb.end())

          rv
        }})

    vds.orderedRVDLeftJoinDistinctAndInsert(nirvanaRVD, "nirvana", product = false)
      .annotateVariantsExpr("vep = va.vep.vep")
  }

  def apply(vsm: MatrixTable, config: String, blockSize: Int = 500000, root: String): MatrixTable =
    annotate(vsm, config, blockSize, root)
}

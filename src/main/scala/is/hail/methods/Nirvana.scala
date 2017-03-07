package is.hail.methods

import java.io.{FileInputStream, IOException}
import java.util.Properties

import is.hail.annotations.Annotation
import is.hail.expr.{JSONAnnotationImpex, Parser, TArray, TBoolean, TDouble, TInt, TString, TStruct}
import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods
import scala.collection.JavaConverters._


object Nirvana {
  val nirvanaWholeSignature = TStruct(
    "header" -> TStruct(
      "annotator" -> TString,
      "creationTime" -> TString,
      "schemaVersion" -> TInt,
      "dataVersion" -> TString,
      "dataSources" -> TArray(TStruct(
        "name" -> TString,
        "version" -> TString,
        "description" -> TString,
        "releaseDate" -> TString
      )),
      "genomeAssembly" -> TString,
      "samples" -> TArray(TString)
    ),
    "positions" -> TArray(TStruct(
      "chromosome" -> TString,
      "refAllele" -> TString,
      "position" -> TInt,
      "altAlleles" -> TArray(TString),
      "cytogeneticBand" -> TString,
      "quality" -> TInt,
      "filters" -> TArray(TString),
      "jointSomaticNormalQuality" -> TInt,
      "copyNumber" -> TInt,
      "strandBias" -> TDouble,
      "recalibratedQuality" -> TDouble,
      "samples" -> TArray(TStruct(
        "variantFreq" -> TDouble,
        "totalDepth" -> TInt,
        "alleleDepths" -> TArray(TInt),
        "genotype" -> TString,
        "genotypeQuality" -> TInt,
        "failedFilter" -> TBoolean,
        "isEmpty" -> TBoolean,
        "copyNumber" -> TInt,
        "lossOfHeterozygosity" -> TBoolean
      )),
      "variants" -> TArray(TStruct(
        "ancestralAllele" -> TString,
        "altAllele" -> TString,
        "refAllele" -> TString,
        "chromosome" -> TString,
        "begin" -> TInt,
        "end" -> TInt,
        "phylopScore" -> TDouble,
        "dbsnp" -> TArray(TString),
        "globalMinorAllele" -> TString,
        "gmaf" -> TDouble,
        "isReferenceMinorAllele" -> TBoolean,
        "variantType" -> TString,
        "vid" -> TString,
        "oneKgAll" -> TDouble,
        "oneKgAllAc" -> TInt,
        "oneKgAllAn" -> TInt,
        "oneKgAfr" -> TDouble,
        "oneKgAfrAc" -> TInt,
        "oneKgAfrAn" -> TInt,
        "oneKgAmr" -> TDouble,
        "oneKgAmrAc" -> TInt,
        "oneKgAmrAn" -> TInt,
        "oneKgEas" -> TDouble,
        "oneKgEasAc" -> TInt,
        "oneKgEasAn" -> TInt,
        "oneKgEur" -> TDouble,
        "oneKgEurAc" -> TInt,
        "oneKgEurAn" -> TInt,
        "oneKgSas" -> TDouble,
        "oneKgSasAc" -> TInt,
        "oneKgSasAn" -> TInt,
        "evsCoverage" -> TInt,
        "evsSamples" -> TInt,
        "evsAll" -> TDouble,
        "evsAfr" -> TDouble,
        "evsEur" -> TDouble,
        "exacCoverage" -> TInt,
        "exacAll" -> TDouble,
        "exacAllAc" -> TInt,
        "exacAllAn" -> TInt,
        "exacAfr" -> TDouble,
        "exacAfrAc" -> TInt,
        "exacAfrAn" -> TInt,
        "exacAmr" -> TDouble,
        "exacAmrAc" -> TInt,
        "exacAmrAn" -> TInt,
        "exacEas" -> TDouble,
        "exacEasAc" -> TInt,
        "exacEasAn" -> TInt,
        "exacFin" -> TDouble,
        "exacFinAc" -> TInt,
        "exacFinAn" -> TInt,
        "exacNfe" -> TDouble,
        "exacNfeAc" -> TInt,
        "exacNfeAn" -> TInt,
        "exacOth" -> TDouble,
        "exacOthAc" -> TInt,
        "exacOthAn" -> TInt,
        "exacSas" -> TDouble,
        "exacSasAc" -> TInt,
        "exacSasAn" -> TInt
      )),
      "regulatoryRegions" -> TArray(TStruct(
        "id" -> TString,
        "consequence" -> TArray(TString)
      )),
      "clinVar" -> TArray(TStruct(
        "id" -> TString,
        "reviewStatus" -> TString,
        "isAlleleSpecific" -> TBoolean,
        "alleleOrigin" -> TString,
        "refAllele" -> TString,
        "altAllele" -> TString,
        "phenotype" -> TString,
        "geneReviewsId" -> TString,
        "medGenId" -> TString,
        "omimId" -> TString,
        "orphanetId" -> TString,
        "significance" -> TString,
        "snoMetCtId" -> TString,
        "lastEvaluatedDate" -> TString,
        "pubMedIds" -> TArray(TString)
      )),
      "cosmic" -> TArray(TStruct(
        "id" -> TString,
        "isAlleleSpecific" -> TBoolean,
        "refAllele" -> TString,
        "altAllele" -> TString,
        "gene" -> TString,
        "studies" -> TArray(TStruct(
          "id" -> TInt,
          "histology" -> TString,
          "primarySite" -> TString
        ))
      )),
      "transcripts" -> TStruct(
        "ensembl" -> TArray(TStruct(
          "transcript" -> TString,
          "aminoAcids" -> TString,
          "bioType" -> TString,
          "cDnaPos" -> TString,
          "codons" -> TString,
          "cdsPos" -> TString,
          "exons" -> TString,
          "introns" -> TString,
          "geneId" -> TString,
          "hgnc" -> TString,
          "consequence" -> TArray(TString),
          "hgvsc" -> TString,
          "hgvsp" -> TString,
          "isCanonical" -> TBoolean,
          "polyPhenScore" -> TDouble,
          "polyPhenPrediction" -> TString,
          "proteinId" -> TString,
          "proteinPos" -> TString,
          "siftScore" -> TDouble,
          "siftPrediction" -> TString
        ))
      )
    ))
  )

  val nirvanaSignature = TStruct(
    "chromosome" -> TString,
    "refAllele" -> TString,
    "position" -> TInt,
    "altAlleles" -> TArray(TString),
    "cytogeneticBand" -> TString,
    "quality" -> TInt,
    "filters" -> TArray(TString),
    "jointSomaticNormalQuality" -> TInt,
    "copyNumber" -> TInt,
    "strandBias" -> TDouble,
    "recalibratedQuality" -> TDouble,
    "samples" -> TArray(TStruct(
      "variantFreq" -> TDouble,
      "totalDepth" -> TInt,
      "alleleDepths" -> TArray(TInt),
      "genotype" -> TString,
      "genotypeQuality" -> TInt,
      "failedFilter" -> TBoolean,
      "isEmpty" -> TBoolean,
      "copyNumber" -> TInt,
      "lossOfHeterozygosity" -> TBoolean
    )),
    "variants" -> TArray(TStruct(
      "ancestralAllele" -> TString,
      "altAllele" -> TString,
      "refAllele" -> TString,
      "chromosome" -> TString,
      "begin" -> TInt,
      "end" -> TInt,
      "phylopScore" -> TDouble,
      "dbsnp" -> TArray(TString),
      "globalMinorAllele" -> TString,
      "gmaf" -> TDouble,
      "isReferenceMinorAllele" -> TBoolean,
      "variantType" -> TString,
      "vid" -> TString,
      "oneKgAll" -> TDouble,
      "oneKgAllAc" -> TInt,
      "oneKgAllAn" -> TInt,
      "oneKgAfr" -> TDouble,
      "oneKgAfrAc" -> TInt,
      "oneKgAfrAn" -> TInt,
      "oneKgAmr" -> TDouble,
      "oneKgAmrAc" -> TInt,
      "oneKgAmrAn" -> TInt,
      "oneKgEas" -> TDouble,
      "oneKgEasAc" -> TInt,
      "oneKgEasAn" -> TInt,
      "oneKgEur" -> TDouble,
      "oneKgEurAc" -> TInt,
      "oneKgEurAn" -> TInt,
      "oneKgSas" -> TDouble,
      "oneKgSasAc" -> TInt,
      "oneKgSasAn" -> TInt,
      "evsCoverage" -> TInt,
      "evsSamples" -> TInt,
      "evsAll" -> TDouble,
      "evsAfr" -> TDouble,
      "evsEur" -> TDouble,
      "exacCoverage" -> TInt,
      "exacAll" -> TDouble,
      "exacAllAc" -> TInt,
      "exacAllAn" -> TInt,
      "exacAfr" -> TDouble,
      "exacAfrAc" -> TInt,
      "exacAfrAn" -> TInt,
      "exacAmr" -> TDouble,
      "exacAmrAc" -> TInt,
      "exacAmrAn" -> TInt,
      "exacEas" -> TDouble,
      "exacEasAc" -> TInt,
      "exacEasAn" -> TInt,
      "exacFin" -> TDouble,
      "exacFinAc" -> TInt,
      "exacFinAn" -> TInt,
      "exacNfe" -> TDouble,
      "exacNfeAc" -> TInt,
      "exacNfeAn" -> TInt,
      "exacOth" -> TDouble,
      "exacOthAc" -> TInt,
      "exacOthAn" -> TInt,
      "exacSas" -> TDouble,
      "exacSasAc" -> TInt,
      "exacSasAn" -> TInt,
      "regulatoryRegions" -> TArray(TStruct(
        "id" -> TString,
        "consequence" -> TArray(TString),
        "type" -> TString
      )),
      "transcripts" -> TStruct(
        "ensembl" -> TArray(TStruct(
          "transcript" -> TString,
          "aminoAcids" -> TString,
          "bioType" -> TString,
          "cDnaPos" -> TString,
          "codons" -> TString,
          "cdsPos" -> TString,
          "exons" -> TString,
          "introns" -> TString,
          "geneId" -> TString,
          "hgnc" -> TString,
          "consequence" -> TArray(TString),
          "hgvsc" -> TString,
          "hgvsp" -> TString,
          "isCanonical" -> TBoolean,
          "polyPhenScore" -> TDouble,
          "polyPhenPrediction" -> TString,
          "proteinId" -> TString,
          "proteinPos" -> TString,
          "siftScore" -> TDouble,
          "siftPrediction" -> TString
        ))
      ),
      "clinVar" -> TArray(TStruct(
        "id" -> TString,
        "reviewStatus" -> TString,
        "isAlleleSpecific" -> TBoolean,
        "alleleOrigin" -> TString,
        "refAllele" -> TString,
        "altAllele" -> TString,
        "phenotype" -> TString,
        "geneReviewsId" -> TString,
        "medGenId" -> TString,
        "omimId" -> TString,
        "orphanetId" -> TString,
        "significance" -> TString,
        "snoMetCtId" -> TString,
        "lastEvaluatedDate" -> TString,
        "pubMedIds" -> TArray(TString)
      )),
      "cosmic" -> TArray(TStruct(
        "id" -> TString,
        "isAlleleSpecific" -> TBoolean,
        "refAllele" -> TString,
        "altAllele" -> TString,
        "gene" -> TString,
        "studies" -> TArray(TStruct(
          "id" -> TInt,
          "histology" -> TString,
          "primarySite" -> TString
        ))
      ))
    ))
  )

  def printContext(w: (String) => Unit) {
    w("##fileformat=VCFv4.1")
    w("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
  }

  def printElement(w: (String) => Unit, v: Variant) {
    val sb = new StringBuilder()
    sb.append(v.contig)
    sb += '\t'
    sb.append(v.start)
    sb.append("\t.\t")
    sb.append(v.ref)
    sb += '\t'
    sb.append(v.altAlleles.iterator.map(_.alt).mkString(","))
    sb.append("\t.\t.\tGT")
    w(sb.result())
  }

  def variantFromInput(contig: String, start: Int, ref: String, altAlleles: Array[String]): Variant = {
    Variant(contig, start, ref, altAlleles)
  }

  def annotate(vds: VariantDataset, config: String, blockSize: Int, root: String = "va.nirvana"): VariantDataset = {
    val parsedRoot = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    val rootType =
      vds.vaSignature.getOption(parsedRoot)
        .filter { t =>
          val r = t == nirvanaSignature
          if (!r) {
            warn(s"type for $parsedRoot does not match Nirvana signature, overwriting.")
          }
          r
        }

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

    val path = properties.getProperty("hail.nirvana.path")

    val cache = properties.getProperty("hail.nirvana.cache")

    val supplementaryAnnotationDirectory = properties.getProperty("hail.nirvana.supplementaryAnnotationDirectory")

    val reference = properties.getProperty("hail.nirvana.reference")

    val rootQuery = rootType
      .map(_ => vds.vaSignature.query(parsedRoot))

    val cmd = Array(
      dotnet,
      s"$nirvanaLocation",
      "-c", cache,
      "--sd", supplementaryAnnotationDirectory,
      "-r", reference,
      "-i", "-",
      "-o", "-"
    )

    val contigQuery = nirvanaSignature.query("chromosome")
    val startQuery = nirvanaSignature.query("position")
    val refQuery = nirvanaSignature.query("refAllele")
    val altsQuery = nirvanaSignature.query("altAlleles")


    val localBlockSize = blockSize

    val annotations = vds.rdd.mapValues { case (va, gs) => va }
      .mapPartitions({ it =>
        val pb = new ProcessBuilder(cmd.toList.asJava)
        val env = pb.environment()

        it.filter { case (v, va) =>
          rootQuery.forall(q => q(va) == null)
        }
          .map { case (v, _) => v }
          .grouped(localBlockSize)
          .flatMap { block =>
            val (jt, proc) = block.iterator.pipe(pb,
              printContext,
              printElement,
              _ => ())
            //Drop header line for now, reconsider later.
            val kt = jt.drop(1).filter(!_.startsWith("]")).map { s =>
                val a = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(s), nirvanaSignature)
              //println(s"a = $a")
                val v = variantFromInput(contigQuery(a).asInstanceOf[String], startQuery(a).asInstanceOf[Int],
                  refQuery(a).asInstanceOf[String], altsQuery(a).asInstanceOf[Seq[String]].toArray)
                (v, a)
              }

            val r = kt.toArray
              .sortBy(_._1)

            val rc = proc.waitFor()
            if (rc != 0)
              fatal(s"nirvana command failed with non-zero exit status $rc")

            r
          }
      }, preservesPartitioning = true)
      .persist(StorageLevel.MEMORY_AND_DISK)


    info(s"nirvana: annotated ${ annotations.count() } variants")

    val (newVASignature, insertNirvana) = vds.vaSignature.insert(nirvanaSignature, parsedRoot)

    val newRDD = vds.rdd
      .zipPartitions(annotations, preservesPartitioning = true) { case (left, right) =>
        left.sortedLeftJoinDistinct(right)
          .map { case (v, ((va, gs), vaNirvana)) =>
            (v, (insertNirvana(va, vaNirvana.orNull), gs))
          }
      }.asOrderedRDD

    vds.copy(rdd = newRDD,
      vaSignature = newVASignature)
  }

}

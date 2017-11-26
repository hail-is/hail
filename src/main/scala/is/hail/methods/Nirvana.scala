package is.hail.methods

import java.io.{FileInputStream, IOException}
import java.util.Properties

import is.hail.annotations.{Annotation, Querier}
import is.hail.expr.{JSONAnnotationImpex, Parser, TArray, TBoolean, TFloat64, TInt32, TSet, TString, TStruct, Type}
import is.hail.utils._
import is.hail.variant.{Locus, Variant, VariantDataset, VariantSampleMatrix}
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

  def annotate(vds: VariantSampleMatrix, config: String, blockSize: Int, root: String = "va.nirvana"): VariantSampleMatrix = {
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

    val path = Option(properties.getProperty("hail.nirvana.path"))

    val cache = properties.getProperty("hail.nirvana.cache")


    val supplementaryAnnotationDirectoryOpt = Option(properties.getProperty("hail.nirvana.supplementaryAnnotationDirectory"))
    val supplementaryAnnotationDirectory = if(supplementaryAnnotationDirectoryOpt.isEmpty) List[String]() else List("--sd", supplementaryAnnotationDirectoryOpt.get)

    val reference = properties.getProperty("hail.nirvana.reference")

    val rootQuery = rootType
      .map(_ => vds.vaSignature.query(parsedRoot))

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
    val oldSignature = vds.vaSignature
    val localBlockSize = blockSize

    info("Running Nirvana")

    val annotations = vds.typedRDD[Locus, Variant, Annotation].mapValues { case (va, gs) => va }
      .mapPartitions({ it =>
        val pb = new ProcessBuilder(cmd.asJava)
        val env = pb.environment()
        if (path.orNull != null)
          env.put("PATH", path.get)

        it.filter { case (v, va) =>
          rootQuery.forall(q => q(va) == null)
        }
          .map { case (v, _) => v }
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
      }, preservesPartitioning = true)
      .persist(StorageLevel.MEMORY_AND_DISK)


    info(s"nirvana: annotated ${ annotations.count() } variants")

    val (newVASignature, insertNirvana) = vds.vaSignature.insert(nirvanaSignature, parsedRoot)

    val newRDD = vds.typedRDD[Locus, Variant, Annotation]
      .zipPartitions(annotations, preservesPartitioning = true) { case (left, right) =>
        left.sortedLeftJoinDistinct(right)
          .map { case (v, ((va, gs), vaNirvana)) =>
            (v, (insertNirvana(va, vaNirvana.orNull), gs))
          }
      }

    vds.copyLegacy(rdd = newRDD,
      vaSignature = newVASignature)
  }
}

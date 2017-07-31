package is.hail.methods

import java.io.{FileInputStream, IOException}
import java.util.Properties

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Locus, Variant, VariantSampleMatrix}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

object VEP {

  val vepSignature = TStruct(
    "assembly_name" -> TString,
    "allele_string" -> TString,
    "ancestral" -> TString,
    "colocated_variants" -> TArray(TStruct(
      "aa_allele" -> TString,
      "aa_maf" -> TDouble,
      "afr_allele" -> TString,
      "afr_maf" -> TDouble,
      "allele_string" -> TString,
      "amr_allele" -> TString,
      "amr_maf" -> TDouble,
      "clin_sig" -> TArray(TString),
      "end" -> TInt,
      "eas_allele" -> TString,
      "eas_maf" -> TDouble,
      "ea_allele" -> TString,
      "ea_maf" -> TDouble,
      "eur_allele" -> TString,
      "eur_maf" -> TDouble,
      "exac_adj_allele" -> TString,
      "exac_adj_maf" -> TDouble,
      "exac_allele" -> TString,
      "exac_afr_allele" -> TString,
      "exac_afr_maf" -> TDouble,
      "exac_amr_allele" -> TString,
      "exac_amr_maf" -> TDouble,
      "exac_eas_allele" -> TString,
      "exac_eas_maf" -> TDouble,
      "exac_fin_allele" -> TString,
      "exac_fin_maf" -> TDouble,
      "exac_maf" -> TDouble,
      "exac_nfe_allele" -> TString,
      "exac_nfe_maf" -> TDouble,
      "exac_oth_allele" -> TString,
      "exac_oth_maf" -> TDouble,
      "exac_sas_allele" -> TString,
      "exac_sas_maf" -> TDouble,
      "id" -> TString,
      "minor_allele" -> TString,
      "minor_allele_freq" -> TDouble,
      "phenotype_or_disease" -> TInt,
      "pubmed" -> TArray(TInt),
      "sas_allele" -> TString,
      "sas_maf" -> TDouble,
      "somatic" -> TInt,
      "start" -> TInt,
      "strand" -> TInt)),
    "context" -> TString,
    "end" -> TInt,
    "id" -> TString,
    "input" -> TString,
    "intergenic_consequences" -> TArray(TStruct(
      "allele_num" -> TInt,
      "consequence_terms" -> TArray(TString),
      "impact" -> TString,
      "minimised" -> TInt,
      "variant_allele" -> TString)),
    "most_severe_consequence" -> TString,
    "motif_feature_consequences" -> TArray(TStruct(
      "allele_num" -> TInt,
      "consequence_terms" -> TArray(TString),
      "high_inf_pos" -> TString,
      "impact" -> TString,
      "minimised" -> TInt,
      "motif_feature_id" -> TString,
      "motif_name" -> TString,
      "motif_pos" -> TInt,
      "motif_score_change" -> TDouble,
      "strand" -> TInt,
      "variant_allele" -> TString)),
    "regulatory_feature_consequences" -> TArray(TStruct(
      "allele_num" -> TInt,
      "biotype" -> TString,
      "consequence_terms" -> TArray(TString),
      "impact" -> TString,
      "minimised" -> TInt,
      "regulatory_feature_id" -> TString,
      "variant_allele" -> TString)),
    "seq_region_name" -> TString,
    "start" -> TInt,
    "strand" -> TInt,
    "transcript_consequences" -> TArray(TStruct(
      "allele_num" -> TInt,
      "amino_acids" -> TString,
      "biotype" -> TString,
      "canonical" -> TInt,
      "ccds" -> TString,
      "cdna_start" -> TInt,
      "cdna_end" -> TInt,
      "cds_end" -> TInt,
      "cds_start" -> TInt,
      "codons" -> TString,
      "consequence_terms" -> TArray(TString),
      "distance" -> TInt,
      "domains" -> TArray(TStruct(
        "db" -> TString,
        "name" -> TString)),
      "exon" -> TString,
      "gene_id" -> TString,
      "gene_pheno" -> TInt,
      "gene_symbol" -> TString,
      "gene_symbol_source" -> TString,
      "hgnc_id" -> TString,
      "hgvsc" -> TString,
      "hgvsp" -> TString,
      "hgvs_offset" -> TInt,
      "impact" -> TString,
      "intron" -> TString,
      "lof" -> TString,
      "lof_flags" -> TString,
      "lof_filter" -> TString,
      "lof_info" -> TString,
      "minimised" -> TInt,
      "polyphen_prediction" -> TString,
      "polyphen_score" -> TDouble,
      "protein_end" -> TInt,
      "protein_start" -> TInt,
      "protein_id" -> TString,
      "sift_prediction" -> TString,
      "sift_score" -> TDouble,
      "strand" -> TInt,
      "swissprot" -> TString,
      "transcript_id" -> TString,
      "trembl" -> TString,
      "uniparc" -> TString,
      "variant_allele" -> TString)),
    "variant_class" -> TString)

  val consequenceIndices = Set(8, 10, 11, 15)

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
    sb.append(v.altAlleles.iterator.map(_.alt).filter(_ != "*").mkString(","))
    sb.append("\t.\t.\tGT")
    w(sb.result())
  }

  def variantFromInput(input: String): Variant = {
    val a = input.split("\t")
    Variant(a(0),
      a(1).toInt,
      a(3),
      a(4).split(","))
  }

  def getCSQHeaderDefinition(cmd: Array[String], perl5lib: String, path: String): Option[String] = {
    val csqHeaderRegex = "ID=CSQ[^>]+Description=\"([^\"]+)".r
    val pb = new ProcessBuilder(cmd.toList.asJava)
    val env = pb.environment()
    if (perl5lib != null)
      env.put("PERL5LIB", perl5lib)
    if (path != null)
      env.put("PATH", path)
    val (jt, proc) = List(Variant("1", 13372, "G", "C")).iterator.pipe(pb,
      printContext,
      printElement,
      _ => ())

    val csqHeader = jt.flatMap(s => csqHeaderRegex.findFirstMatchIn(s).map(m => m.group(1)))
    val rc = proc.waitFor()
    if (rc != 0)
      fatal(s"vep command failed with non-zero exit status $rc")

    if (csqHeader.hasNext)
      Some(csqHeader.next())
    else {
      warn("Could not get VEP CSQ header")
      None
    }
  }

  def getNonStarToOriginalAlleleIdxMap(v: Variant): mutable.Map[Int, Int] = {
    val alleleMap = mutable.Map[Int, Int]()
    (0 until v.nAltAlleles).foldLeft(0) {
      case (nStar, aai) =>
        if (v.altAlleles(aai).alt == "*")
          nStar + 1
        else {
          alleleMap(aai - nStar + 1) = aai + 1
          nStar
        }
    }
    alleleMap
  }

  def annotate[T >: Null](vsm: VariantSampleMatrix[Locus, Variant, T], config: String, root: String = "va.vep", csq: Boolean,
    blockSize: Int)(implicit tct: ClassTag[T]): VariantSampleMatrix[Locus, Variant, T] = {

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

    val perl = properties.getProperty("hail.vep.perl", "perl")

    val perl5lib = properties.getProperty("hail.vep.perl5lib")

    val path = properties.getProperty("hail.vep.path")

    val location = properties.getProperty("hail.vep.location")
    if (location == null)
      fatal("property `hail.vep.location' required")

    val cacheDir = properties.getProperty("hail.vep.cache_dir")
    if (cacheDir == null)
      fatal("property `hail.vep.cache_dir' required")


    val plugin = if (properties.getProperty("hail.vep.plugin") != null) {
         properties.getProperty("hail.vep.plugin")
    } else {

        val humanAncestor = properties.getProperty("hail.vep.lof.human_ancestor")
        if (humanAncestor == null)
          fatal("property `hail.vep.lof.human_ancestor' required")

        val conservationFile = properties.getProperty("hail.vep.lof.conservation_file")
        if (conservationFile == null)
          fatal("property `hail.vep.lof.conservation_file' required")

        s"LoF,human_ancestor_fa:$humanAncestor,filter_position:0.05,min_intron_size:15,conservation_file:$conservationFile"
    }

    val fasta = properties.getProperty("hail.vep.fasta")
    if (fasta == null)
      fatal("property `hail.vep.fasta' required")

    var assembly = properties.getProperty("hail.vep.assembly")
    if (assembly == null) {
      warn("property `hail.vep.assembly' not specified. Setting to GRCh37")
      assembly = "GRCh37"
    }

    val cmd =
      Array(
        perl,
        s"$location",
        "--format", "vcf",
        if (csq) "--vcf" else "--json",
        "--everything",
        "--allele_number",
        "--no_stats",
        "--cache", "--offline",
        "--dir", s"$cacheDir",
        "--fasta", s"$fasta",
        "--minimal",
        "--assembly", s"$assembly",
        "--plugin", s"$plugin",
        "-o", "STDOUT")

    val inputQuery = vepSignature.query("input")

    val csqRegex = "CSQ=[^;^\\t]+".r

    val localBlockSize = blockSize

    val csqHeader = if (csq) getCSQHeaderDefinition(cmd, perl5lib, path).getOrElse("") else ""
    val alleleNumIndex = if (csq) csqHeader.split("\\|").indexOf("ALLELE_NUM") else -1

    val annotations = vsm.rdd
      .mapPartitions({ it =>
        val pb = new ProcessBuilder(cmd.toList.asJava)
        val env = pb.environment()
        if (perl5lib != null)
          env.put("PERL5LIB", perl5lib)
        if (path != null)
          env.put("PATH", path)

        it
          .grouped(localBlockSize)
          .flatMap { block =>
            val (jt, proc) = block.iterator.map { case (v, (va, gs)) => v }.pipe(pb,
              printContext,
              printElement,
              _ => ())

            val nonStarToOriginalVariant = block.map { case (v, (va, gs)) =>
              (v.copy(altAlleles = v.altAlleles.filter(_.alt != "*")), v)
            }.toMap

            val kt = jt
              .filter(s => !s.isEmpty && s(0) != '#')
              .map { s =>
                if (csq) {
                  val vvep = variantFromInput(s)
                  if (!nonStarToOriginalVariant.contains(vvep))
                    fatal(s"VEP output variant ${ vvep } not found in original variants.\nVEP output: $s")

                  val v = nonStarToOriginalVariant(vvep)
                  val x = csqRegex.findFirstIn(s)
                  x match {
                    case Some(value) =>
                      val tr_aa = value.substring(4).split(",")
                      if (vvep != v && alleleNumIndex > -1) {
                        val alleleMap = getNonStarToOriginalAlleleIdxMap(v)
                        (v, tr_aa.map {
                          x =>
                            val xsplit = x.split("\\|")
                            val allele_num = xsplit(alleleNumIndex)
                            if (allele_num.isEmpty)
                              x
                            else
                              xsplit
                                .updated(alleleNumIndex, alleleMap.getOrElse(xsplit(alleleNumIndex).toInt, "").toString)
                                .mkString("|")
                        }: IndexedSeq[Annotation])
                      } else
                        (v, tr_aa: IndexedSeq[Annotation])
                    case None =>
                      warn(s"No VEP annotation found for variant $v. VEP returned $s.")
                      (v, IndexedSeq.empty[Annotation])
                  }
                } else {
                  val a = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(s), vepSignature)
                  val vvep = variantFromInput(inputQuery(a).asInstanceOf[String])

                  if (!nonStarToOriginalVariant.contains(vvep))
                    fatal(s"VEP output variant ${ vvep } not found in original variants.\nVEP output: $s")

                  val v = nonStarToOriginalVariant(vvep)
                  if (vvep != v) {
                    val alleleMap = getNonStarToOriginalAlleleIdxMap(v)
                    (v, consequenceIndices.foldLeft(a.asInstanceOf[Row]) {
                      case (r, i) =>
                        if (r(i) == null)
                          r
                        else {
                          r.update(i, r(i).asInstanceOf[IndexedSeq[Row]].map {
                            x => x.update(0, alleleMap.getOrElse(x.getInt(0), null))
                          })
                        }
                    })
                  } else
                    (v, a)
                }
              }

            val r = kt.toArray
              .sortBy(_._1)

            val rc = proc.waitFor()
            if (rc != 0)
              fatal(s"vep command '${cmd.mkString(" ")}' failed with non-zero exit status $rc")

            r
          }
      }, preservesPartitioning = true)
      .persist(StorageLevel.MEMORY_AND_DISK)

    info(s"vep: annotated ${ annotations.count() } variants")

    val (newVASignature, insertVEP) = vsm.vaSignature.insert(if (csq) TArray(TString) else vepSignature, parsedRoot)

    val newRDD = vsm.rdd
      .zipPartitions(annotations, preservesPartitioning = true) { case (left, right) =>
        new Iterator[(Variant, (Annotation, Iterable[T]))] {
          def hasNext: Boolean = {
            val r = left.hasNext
            assert(r == right.hasNext)
            r
          }

          def next(): (Variant, (Annotation, Iterable[T])) = {
            val (lv, (va, gs)) = left.next()
            val (rv, vaVep) = right.next()
            assert(lv == rv)
            (lv, (insertVEP(va, vaVep), gs))
          }
        }
      }.asOrderedRDD

    (csq, newVASignature) match {
      case (true, t: TStruct) => vsm.copy(rdd = newRDD,
        vaSignature = t.setFieldAttributes(parsedRoot, Map("Description" -> csqHeader)))
      case _ => vsm.copy(rdd = newRDD, vaSignature = newVASignature)
    }
  }
}

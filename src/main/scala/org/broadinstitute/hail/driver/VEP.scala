package org.broadinstitute.hail.driver

import java.io.{FileInputStream, IOException}
import java.util.Properties

import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Variant
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object VEP extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "--config", usage = "VEP configuration file")
    var config: String = _

    @Args4jOption(name = "-r", aliases = Array("--root"), usage = "Variant annotation path to store VEP output")
    var root: String = "va.vep"
  }

  def newOptions = new Options

  def name = "vep"

  def description = "Annotation variants with VEP"

  def supportsMultiallelic = true

  def requiresVDS = true

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
      "hgnc_id" -> TInt,
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

  def printContext(w: (String) => Unit) {
    w("##fileformat=VCFv4.1")
    w("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
  }

  def printElement(v: Variant, w: (String) => Unit) {
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

  def variantFromInput(input: String): Variant = {
    val a = input.split("\t")
    Variant(a(0),
      a(1).toInt,
      a(3),
      a(4).split(","))
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val root = options.root.split("\\.").toList
    if (root.isEmpty
      || root.head != "va")
      fatal("root must begin with `va.'")

    val properties = try {
      val p = new Properties()
      val is = new FileInputStream(options.config)
      p.load(is)
      is.close()
      p
    } catch {
      case e: IOException =>
        fatal(s"could not open file: ${e.getMessage}")
    }

    val perl = properties.getProperty("hail.vep.perl", "perl")

    val env = mutable.Map.empty[String, String]
    val perl5lib = properties.getProperty("hail.vep.perl5lib")
    if (perl5lib != null)
      env += ("PERL5LIB" -> perl5lib)

    val path = properties.getProperty("hail.vep.path")
    if (path != null)
      env += ("PATH" -> path)

    val location = properties.getProperty("hail.vep.location")
    if (location == null)
      fatal("property `hail.vep.location' required")

    val cacheDir = properties.getProperty("hail.vep.cache_dir")
    if (location == null)
      fatal("property `hail.vep.cache_dir' required")

    val humanAncestor = properties.getProperty("hail.vep.lof.human_ancestor")
    if (location == null)
      fatal("property `hail.vep.human_ancestor' required")

    val conservationFile = properties.getProperty("hail.vep.lof.conservation_file")
    if (conservationFile == null)
      fatal("property `hail.vep.conservation_file' required")

    val cmd = Array(
      perl,
      s"${location}",
      "--format", "vcf",
      "--json",
      "--everything",
      "--allele_number",
      "--no_stats",
      "--cache", "--offline",
      "--dir", s"${cacheDir}",
      "--fasta", s"${cacheDir}/homo_sapiens/81_GRCh37/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa",
      "--minimal",
      "--assembly", "GRCh37",
      "--plugin", s"LoF,human_ancestor_fa:${humanAncestor},filter_position:0.05,min_intron_size:15,conservation_file:${conservationFile}",
      "-o", "STDOUT")

    log.info(s"vep env: ${env.map { case (k, v) => s"$k=$v" }.mkString(";")}")
    log.info(s"vep command: ${cmd.mkString(" ")}")

    val inputQuery = vepSignature.query("input")

    val annotations = vds.rdd
      .map { case (v, va, gs) => v }
      .pipe(cmd,
        env,
        printContext,
        printElement)
      .map { jv =>
        val a = Annotation.fromJson(parse(jv), vepSignature, "<root>")
        val v = variantFromInput(inputQuery(a).get.asInstanceOf[String])
        (v, a)
      }.persist(StorageLevel.MEMORY_AND_DISK)

    val (newVASignature, insertVEP) = vds.vaSignature.insert(vepSignature, root.tail)

    val newRDD = vds.rdd
      .zipPartitions(annotations) { case (it, ita) =>
        val its = it.toArray.sortWith { case ((v1, _, _), (v2, _, _)) => v1 < v2 }
        val itas = ita.toArray.sortWith { case ((v1, _), (v2, _)) => v1 < v2 }
        its.iterator.zip(itas.iterator)
          .map { case ((v1, va, gs), (v2, vep)) =>
            assert(v1 == v2)
            (v1, insertVEP(va, Some(vep)), gs)
          }
      }

    val newVDS = vds.copy(rdd = newRDD,
      vaSignature = newVASignature)

    state.copy(vds = newVDS)
  }
}

package org.broadinstitute.hail.driver

import java.io.{IOException, FileInputStream, InputStream}
import java.util.Properties
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.Variant
import org.json4s._
import org.json4s.native.JsonMethods._
import scala.collection.mutable
import org.kohsuke.args4j.{Option => Args4jOption}

object VEP extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "--config", usage = "VEP configuration file")
    var config: String = _
  }

  def newOptions = new Options

  def name = "vep"

  def description = "Annotation variants with VEP"

  override def supportsMultiallelic = true

  def jsonToAnnotation(jv: JValue, t: Type, parent: String): Any = (jv, t) match {
    case (JNull, _) => null
    case (JInt(x), TInt) => x.toInt
    case (JInt(x), TDouble) => x.toDouble
    case (JInt(x), TString) => x.toString
    case (JLong(x), TLong) => x
    case (JDouble(x), TDouble) => x
    case (JString(x), TString) => x
    case (JString(x), TDouble) =>
      if (x.startsWith("-:"))
        x.drop(2).toDouble
      else
        x.toDouble
    case (JBool(x), TBoolean) => x

    case (JObject(jfields), TStruct(fields)) =>
      val m = mutable.Map.empty[String, Any]
      for ((name, jv2) <- jfields) {
        fields.get(name) match {
          case Some(f) =>
            val v2 = jsonToAnnotation(jv2, f.`type`, parent + "." + name)
            if (v2 != null)
              m += ((name, v2))

          case None =>
            warn(s"Signature for $parent has no field $name")
        }
      }
      Annotations(m.toMap)

    case (JArray(a), TArray(elementType)) =>
      a.iterator.map(jv2 => jsonToAnnotation(jv2, elementType, parent + ".<array>")).toArray[Any]: IndexedSeq[Any]

    case _ =>
      warn(s"Can't convert json value $jv to signature $t for $parent.")
      null
  }

  val vepSignature = TStruct(
    "assembly_name" -> TString,
    "id" -> TString,
    "input" -> TString,
    "allele_string" -> TString,
    "seq_region_name" -> TString,
    "start" -> TInt,
    "end" -> TInt,
    "strand" -> TInt,
    "variant_class" -> TString,
    "most_severe_consequence" -> TString,
    "motif_feature_consequences" -> TArray(TStruct(
      "motif_name" -> TString,
      "minimised" -> TInt,
      "high_inf_pos" -> TString,
      "strand" -> TInt,
      "motif_feature_id" -> TString,
      "consequence_terms" -> TArray(TString),
      "variant_allele" -> TString,
      "impact" -> TString,
      "motif_pos" -> TInt,
      "motif_score_change" -> TDouble,
      "allele_num" -> TInt)),
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
    "colocated_variants" -> TArray(TStruct(
      "clin_sig" -> TArray(TString),
      "pubmed" -> TArray(TInt),
      "end" -> TInt,
      "aa_allele" -> TString,
      "exac_eas_allele" -> TString,
      "exac_nfe_allele" -> TString,
      "allele_string" -> TString,
      "start" -> TInt,
      "eas_allele" -> TString,
      "aa_maf" -> TDouble,
      "exac_adj_allele" -> TString,
      "eur_allele" -> TString,
      "amr_maf" -> TDouble,
      "exac_sas_allele" -> TString,
      "exac_amr_maf" -> TDouble,
      "eur_maf" -> TDouble,
      "exac_nfe_maf" -> TDouble,
      "exac_adj_maf" -> TDouble,
      "exac_amr_allele" -> TString,
      "sas_maf" -> TDouble,
      "exac_oth_maf" -> TDouble,
      "exac_allele" -> TString,
      "afr_maf" -> TDouble,
      "exac_fin_maf" -> TDouble,
      "exac_fin_allele" -> TString,
      "exac_oth_allele" -> TString,
      "minor_allele" -> TString,
      "exac_afr_allele" -> TString,
      "minor_allele_freq" -> TDouble,
      "exac_maf" -> TDouble,
      "afr_allele" -> TString,
      "strand" -> TInt,
      "sas_allele" -> TString,
      "eas_maf" -> TDouble,
      "exac_eas_maf" -> TDouble,
      "ea_allele" -> TString,
      "ea_maf" -> TDouble,
      "id" -> TString,
      "exac_sas_maf" -> TDouble,
      "amr_allele" -> TString,
      "exac_afr_maf" -> TDouble,
      "phenotype_or_disease" -> TInt,
      "somatic" -> TInt)),
    "regulatory_feature_consequences" -> TArray(TStruct(
      "allele_num" -> TInt,
      "biotype" -> TString,
      "consequence_terms" -> TArray(TString),
      "impact" -> TString,
      "minimised" -> TInt,
      "regulatory_feature_id" -> TString,
      "variant_allele" -> TString)),
    "intergenic_consequences" -> TArray(TStruct(
      "allele_num" -> TInt,
      "consequence_terms" -> TArray(TString),
      "impact" -> TString,
      "minimised" -> TInt,
      "variant_allele" -> TString)))

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

    val location = properties.getProperty("hail.vep.location")
    if (location == null)
      fatal("property `hail.vep.location' required")

    val cacheDir = properties.getProperty("hail.vep.cache_dir")
    if (location == null)
      fatal("property `hail.vep.cache_dir' required")

    val humanAncestor = properties.getProperty("hail.vep.human_ancestor")
    if (location == null)
      fatal("property `hail.vep.human_ancestor' required")

    val conservationFile = properties.getProperty("hail.vep.conservation_file")
    if (conservationFile == null)
      fatal("property `hail.vep.conservation_file' required")

    val annotations = vds.rdd
      .map { case (v, va, gs) => v }
      .pipe(Array(
        s"${location}",
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
        "-o", "STDOUT"),
        Map.empty,
        printContext,
        printElement)
      .map { jv =>
        val a = jsonToAnnotation(parse(jv), vepSignature, "<root>").asInstanceOf[Annotations]
        val v = variantFromInput(a.get("input"))
        (v, a)
      }.persist(StorageLevel.MEMORY_AND_DISK)

    val newRDD = vds.rdd
      .zipPartitions(annotations) { case (it, ita) =>
        val its = it.toArray.sortWith { case ((v1, _, _), (v2, _, _)) => v1 < v2 }
        val itas = ita.toArray.sortWith { case ((v1, _), (v2, _)) => v1 < v2 }
        its.iterator.zip(itas.iterator)
          .map { case ((v1, va, gs), (v2, vep)) =>
            assert(v1 == v2)
            (v1, va ++ Annotations(Map("vep" -> vep)), gs)
          }
      }

    val newVDS = vds.copy(rdd = newRDD,
      metadata = vds.metadata.addVariantAnnotationSignatures("vep", vepSignature))

    state.copy(vds = newVDS)
  }
}

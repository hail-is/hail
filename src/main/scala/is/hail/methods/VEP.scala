package is.hail.methods

import java.io.{FileInputStream, IOException}
import java.util.Properties

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods
import scala.collection.JavaConverters._

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

  def variantFromInput(input: String): Variant = {
    val a = input.split("\t")
    Variant(a(0),
      a(1).toInt,
      a(3),
      a(4).split(","))
  }

  def annotate(vds: VariantDataset, config: String, root: String = "va.vep", csq: Boolean,
    force: Boolean, blockSize: Int): VariantDataset = {

    val parsedRoot = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    val rootType =
      vds.vaSignature.getOption(parsedRoot)
        .filter { t =>
          val r = t == (if(csq) TString else vepSignature)
          if (!r) {
            if (force)
              warn(s"type for $parsedRoot does not match vep signature, overwriting.")
            else
              warn(s"type for $parsedRoot does not match vep signature.")
          }
          r
        }

    if (rootType.isEmpty && !force)
      fatal("for performance, you should annotate variants with pre-computed VEP annotations.  Cowardly refusing to VEP annotate from scratch.  Use --force to override.")

    val rootQuery = rootType
      .map(_ => vds.vaSignature.query(parsedRoot))

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

    val humanAncestor = properties.getProperty("hail.vep.lof.human_ancestor")
    if (humanAncestor == null)
      fatal("property `hail.vep.human_ancestor' required")

    val conservationFile = properties.getProperty("hail.vep.lof.conservation_file")
    if (conservationFile == null)
      fatal("property `hail.vep.conservation_file' required")

    val cmd =
      Array(
      perl,
      s"$location",
      "--format", "vcf",
        if(csq) "--vcf" else "--json",
      "--everything",
      "--allele_number",
      "--no_stats",
      "--cache", "--offline",
      "--dir", s"$cacheDir",
      "--fasta", s"$cacheDir/homo_sapiens/81_GRCh37/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa",
      "--minimal",
      "--assembly", "GRCh37",
      "--plugin", s"LoF,human_ancestor_fa:$humanAncestor,filter_position:0.05,min_intron_size:15,conservation_file:$conservationFile",
      "-o", "STDOUT")

    val inputQuery = vepSignature.query("input")

    val csq_regex = "CSQ=[^;^\\t]+".r

    val localBlockSize = blockSize

    val annotations = vds.rdd.mapValues { case (va, gs) => va }
      .mapPartitions({ it =>
        val pb = new ProcessBuilder(cmd.toList.asJava)
        val env = pb.environment()
        if (perl5lib != null)
          env.put("PERL5LIB", perl5lib)
        if (path != null)
          env.put("PATH", path)

        it.filter { case (v, va) =>
          rootQuery.flatMap(q => q(va)).isEmpty
        }
          .map { case (v, _) => v }
          .grouped(localBlockSize)
          .flatMap(_.iterator.pipe(pb,
            printContext,
            printElement,
            _ => ())
            .map { s =>
              if(csq) {
                csq_regex.findFirstIn(s).map(x => (variantFromInput(s), x.substring(4)))
              }else{
                val a = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(s), vepSignature)
                val v = variantFromInput(inputQuery(a).get.asInstanceOf[String])
                Option((v, a))
              }
            }
            .flatten
            .toArray
            .sortBy(_._1))
      }, preservesPartitioning = true)
      .persist(StorageLevel.MEMORY_AND_DISK)

    info(s"vep: annotated ${ annotations.count() } variants")

    val (newVASignature, insertVEP) = vds.vaSignature.insert( if(csq) TString else vepSignature, parsedRoot)

    val newRDD = vds.rdd
      .zipPartitions(annotations, preservesPartitioning = true) { case (left, right) =>
        left.sortedLeftJoinDistinct(right)
          .map { case (v, ((va, gs), vaVep)) =>
            (v, (vaVep.map(a => insertVEP(va, Some(a))).getOrElse(va), gs))
          }
      }.asOrderedRDD

    vds.copy(rdd = newRDD,
      vaSignature = newVASignature)
  }
}

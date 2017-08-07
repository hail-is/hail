package is.hail.io.vcf

import com.intel.genomicsdb.GenomicsDBFeatureReader
import htsjdk.variant.vcf.VCFHeader
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import is.hail.utils._
import is.hail.variant.{VSMLocalValue, VSMMetadata, VariantSampleMatrix, Locus, Variant}
import org.apache.spark.storage.StorageLevel
import org.json4s._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters.asScalaIteratorConverter
import java.io.{FileWriter, File}


case class QueryJSON(workspace: String,
                     array: String,
                     vid_mapping_file: String,
                     callset_mapping_file: String,
                     vcf_header_filename: Option[String],
                     reference_genome: String,
                     query_column_ranges: List[List[List[Int]]] = List(List(List(0, 17421565))), //TODO: fix this to match gdb docs description
                     query_attributes: List[String] = List(),
                     query_row_ranges: Option[List[List[Int]]] = None,
                     max_diploid_alt_alleles_that_can_be_genotyped: Option[Int] = None,
                     vcf_output_filename: Option[List[String]] = None,
                     vcf_output_format: Option[String] = None,
                     produce_GT_field: Boolean = true,
                     index_output_VCF: Option[Boolean] = None,
                     combined_vcf_records_buffer_size_limit: Option[Int] = None)

object LoadGDB {

  def createQueryJSON(tiledbworkspace: String,
                      arrayName: String,
                      vid_mapping_file: String,
                      callsets_mapping_file: String,
                      vcfHeaderPath: Option[String], //might not need this b/c won't be generating vcf headers from given gdb arrays in the future
                      refGenome: String): File = {
    val tempFile = File.createTempFile("sample2query", ".json")
    jackson.Serialization.writePretty(QueryJSON(tiledbworkspace,
      arrayName,
      vid_mapping_file,
      callsets_mapping_file,
      vcfHeaderPath,
      refGenome),
      new FileWriter(tempFile))
    tempFile.getCanonicalFile
  }

  /* PATH PARAMETERS REQUIRE ABSOLUTE PATHS */
  def apply[T >: Null](hc: HailContext,
               reader: HtsjdkRecordReader,
               loaderJSONFile: String,
               tiledbWorkspace: String,
               arrayName: String,
               vid_mapping_file: String,
               callsets_mapping_file: String,
               vcfHeaderPath: Option[String],
               refGenome: String,
               nPartitions: Option[Int] = None,
               dropSamples: Boolean = false): VariantSampleMatrix[Locus, Variant, Annotation] = {
    val sc = hc.sc

    val codec = new htsjdk.variant.vcf.VCFCodec()

    val queryFile = createQueryJSON(tiledbWorkspace,
      arrayName,
      vid_mapping_file,
      callsets_mapping_file,
      vcfHeaderPath,
      refGenome)

    val gdbReader = new GenomicsDBFeatureReader(loaderJSONFile, queryFile.getCanonicalPath, codec)

    val header = gdbReader
      .getHeader
      .asInstanceOf[VCFHeader]

    // FIXME apply descriptions when HTSJDK is fixed to expose filter descriptions
    val immutableFilters: Map[String, String] = header
      .getFilterLines
      .toList
      //(ID, description)
      .map(line => (line.getID, ""))
      .toMap

    var filters = immutableFilters

    if (immutableFilters.size > 1 && immutableFilters.contains("PASS")) { //remove extra PASS filter if there are others
      val mutableFilters = collection.mutable.Map(immutableFilters.toSeq: _*)
      mutableFilters.remove("PASS")
      filters = mutableFilters.toMap
    }

    val infoHeader = header.getInfoHeaderLines
    val infoSignature = LoadVCF.headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val genotypeSignature: Type = {
      val callFields = reader.callFields
      LoadVCF.headerSignature(formatHeader, callFields).getOrElse(TStruct.empty)
    }

    val variantAnnotationSignatures = TStruct(
      Array(
        Some(Field("rsid", TString, 0)),
        Some(Field("qual", TFloat64, 1)),
        Some(Field("filters", TSet(TString), 2, filters)),
        infoSignature.map(sig => Field("info", sig, 3))
      ).flatten)

    val sampleIds: Array[String] =
      if (dropSamples)
        Array.empty
      else {
        val it = gdbReader.query("", 0, 1).asScala
        if (it.hasNext)
          it.next.getSampleNamesOrderedByName.toArray(new Array[String](0))
        else
          Array.empty
      }

    val records = gdbReader
      .iterator
      .asScala
      .map(vc => {
        reader.readRecord(vc, infoSignature, genotypeSignature)
      })
      .toSeq

    val recordRDD = sc.parallelize(records, nPartitions.getOrElse(sc.defaultMinPartitions))

    val justVariants = recordRDD.map(_._1)
    justVariants.persist(StorageLevel.MEMORY_AND_DISK)

    val noMulti = justVariants.forall(_.nAlleles == 2)

    if (noMulti)
      info("No multiallelics detected.")
    else
      info("Multiallelic variants detected. Some methods require splitting or filtering multiallelics first.")

    val rdd = recordRDD.toOrderedRDD(justVariants)

    justVariants.unpersist()
    queryFile.delete()

    new VariantSampleMatrix(hc, VSMMetadata(
      TString,
      TStruct.empty,
      TVariant,
      variantAnnotationSignatures,
      TStruct.empty,
      genotypeSignature,
      wasSplit = noMulti),
      VSMLocalValue(Annotation.empty,
        sampleIds,
        Annotation.emptyIndexedSeq(sampleIds.length)),
      rdd)
  }
}

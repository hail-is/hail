package is.hail.io.vcf

import com.intel.genomicsdb.GenomicsDBFeatureReader
import htsjdk.variant.vcf.{VCFCompoundHeaderLine, VCFHeader}
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{GenomeReference, MatrixTable, Variant}
import org.json4s._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters.asScalaIteratorConverter
import java.io.{File, FileWriter}

import is.hail.expr.types._
import is.hail.io.VCFAttributes
import is.hail.io.vcf.LoadVCF.headerSignature

import scala.collection.mutable

case class QueryJSON(workspace: String,
                     array: String,
                     vid_mapping_file: String,
                     callset_mapping_file: String,
                     vcf_header_filename: Option[String],
                     //type List[List[Any]] b/c genomicsDB accepts both single columns and ranges of columns to query
                     query_column_ranges: List[List[Any]] = List(List(List(0, 2829728720l))),
                     reference_genome: String = "",
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
                      vcfHeaderPath: Option[String]): File = {
    val tempFile = File.createTempFile("sample2query", ".json")
    jackson.Serialization.write(QueryJSON(tiledbworkspace,
      arrayName,
      vid_mapping_file,
      callsets_mapping_file,
      vcfHeaderPath),
      new FileWriter(tempFile))
    tempFile.getCanonicalFile
  }

  def formatHeaderSignature[T <: VCFCompoundHeaderLine](lines: java.util.Collection[T],
    callFields: Set[String] = Set.empty[String], arrayElementsRequired: Boolean = true): (TStruct, Int, VCFAttributes) = {
    val canonicalFields = Array(
      "GT" -> TCall(),
      "AD" -> TArray(TInt32(arrayElementsRequired)),
      "DP" -> TInt32(),
      "GQ" -> TInt32(),
      "PL" -> TArray(TInt32(arrayElementsRequired)))

    val (raw, attrs) = headerSignature(lines, callFields, arrayElementsRequired)

    var canonicalFlags = 0
    var i = 0
    val done = mutable.Set[Int]()
    val fb = new ArrayBuilder[Field]()
    canonicalFields.zipWithIndex.foreach { case ((id, t), j) =>
      if (raw.hasField(id)) {
        val f = raw.field(id)
        if (f.typ == t) {
          done += f.index
          fb += Field(f.name, f.typ, i)
          canonicalFlags |= (1 << j)
          i += 1
        }
      }
    }

    raw.fields.foreach { f =>
      if (!done.contains(f.index)) {
        fb += Field(f.name, f.typ, i)
        i += 1
      }
    }

    (TStruct(fb.result()), canonicalFlags, attrs)
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
               nPartitions: Option[Int] = None,
               dropSamples: Boolean = false,
               gr: GenomeReference = GenomeReference.defaultReference): MatrixTable = {
    val sc = hc.sc

    val codec = new htsjdk.variant.vcf.VCFCodec()

    val queryFile = createQueryJSON(tiledbWorkspace,
      arrayName,
      vid_mapping_file,
      callsets_mapping_file,
      vcfHeaderPath)

    val gdbReader = new GenomicsDBFeatureReader(loaderJSONFile, queryFile.getCanonicalPath, codec)

    val header = gdbReader
      .getHeader
      .asInstanceOf[VCFHeader]

    // FIXME apply descriptions when HTSJDK is fixed to expose filter descriptions
    val immutableFilters: VCFAttributes = header
      .getFilterLines
      .toList
      //(ID, description)
      .map(line => (line.getID, Map("Description" -> "")))
      .toMap

    var filters = immutableFilters

    if (immutableFilters.size > 1 && immutableFilters.contains("PASS")) { //remove extra PASS filter if there are others
      val mutableFilters = collection.mutable.Map(immutableFilters.toSeq: _*)
      mutableFilters.remove("PASS")
      filters = mutableFilters.toMap
    }

    val infoHeader = header.getInfoHeaderLines
    val (infoSignature, infoAttrs) = LoadVCF.headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val (genotypeSignature, canonicalFlags, formatAttrs) = formatHeaderSignature(formatHeader, reader.callFields)

    val variantAnnotationSignatures = TStruct(Array(
      Field("rsid", TString(), 0),
      Field("qual", TFloat64(), 1),
      Field("filters", TSet(TString()), 2),
      Field("info", infoSignature, 3)))

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

    val rowType = TStruct(
      "v" -> TVariant(gr),
      "va" -> variantAnnotationSignatures,
      "gs" -> TArray(genotypeSignature))

    val localRowType = rowType

    val region = Region()
    val rvb = new RegionValueBuilder(region)

    val records = gdbReader
      .iterator
      .asScala
      .map { vc =>
        region.clear()
        rvb.start(rowType.fundamentalType)
        reader.readRecord(vc, rvb, infoSignature, genotypeSignature, dropSamples, canonicalFlags)

        val ur = new UnsafeRow(localRowType, region.copy(), rvb.end())

        val v = ur.getAs[Variant](0)
        gr.checkVariant(v)

        val va = ur.get(1)
        val gs: Iterable[Annotation] = ur.getAs[IndexedSeq[Annotation]](2)

        (v: Annotation, (va, gs: Iterable[Annotation]))
      }
      .toSeq

    val recordRDD = sc.parallelize(records, nPartitions.getOrElse(sc.defaultMinPartitions))

    queryFile.delete()

    MatrixTable.fromLegacy(hc, MatrixType(
      colType = TStruct("s" -> TString()),
      colKey = Array("s"),
      vType = TVariant(gr),
      vaType = variantAnnotationSignatures,
      genotypeType = genotypeSignature),
      Annotation.empty,
      sampleIds.map(x => Annotation(x)),
      recordRDD)
  }
}

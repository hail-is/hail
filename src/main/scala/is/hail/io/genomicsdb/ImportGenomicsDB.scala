package is.hail.io.genomicsdb

import java.io.File

import com.intel.genomicsdb.GenomicsDBFeatureReader
import htsjdk.variant.bcf2.BCF2Codec
import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.vcf.{VCFFormatHeaderLine, VCFHeader}
import is.hail.utils._
import is.hail.HailContext
import is.hail.annotations.{BroadcastIndexedSeq, Region, RegionValue, RegionValueBuilder}
import is.hail.expr.{MatrixImportGenomicsDB, Parser}
import is.hail.expr.types._
import is.hail.io.VCFAttributes
import is.hail.io.vcf.{BufferedLineIterator, HtsjdkRecordReader, LoadVCF}
import is.hail.utils.Interval
import is.hail.variant.{MatrixTable, ReferenceGenome}
import org.apache.hadoop
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.Row
import org.json4s.JObject
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods.parse

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.sys.process._
import scala.language.postfixOps

case class GenomicsDBShard(
  interval: Interval,
  filename: String)


case class GenomicsDBFileMetadata(
  sample_names: Array[String],
  shards: Array[JObject],
  vcfheader: String)

case class GenomicsDBMetadata(
  rg: ReferenceGenome,
  fastaFilename: String,
  baseDirname: String,
  sampleIds: Array[String],
  typ: MatrixType,
  infoType: TStruct,
  callFields: Set[String],
  canonicalFlags: Int,
  infoFlagFieldNames: Set[String],
  shards: Array[GenomicsDBShard])

object ImportGenomicsDB {
  val DEFAULT_ARRAY_NAME: String = "genomicsdb_array"
  val DEFAULT_VIDMAP_FILE_NAME: String = "vidmap.json"
  val DEFAULT_CALLSETMAP_FILE_NAME: String = "callset.json"
  val DEFAULT_VCFHEADER_FILE_NAME: String = "vcfheader.vcf"

  def readShard(hadoopConf: hadoop.conf.Configuration, metadata: GenomicsDBMetadata, typ: MatrixType, i: Int): Iterator[RegionValue] = {
    val localShardTmpdir = hadoopConf.getTemporaryFile("file:///tmp", prefix = Some("genomicsdb-shard-"))
    hadoopConf.mkDir(localShardTmpdir)

    val shardFilename = metadata.shards(i).filename
    val shardPath = new Path(shardFilename)
    val shardName = shardPath.getName
    val shard = metadata.baseDirname + "/" + shardFilename

    val localShard = localShardTmpdir + "/" + shardName

    log.info(s"localizing GenomicsDB shard: $shard => $localShard")
    hadoopConf.copy(shard, localShard)

    val untarCommand = Seq("bash", "-c", s"(cd ${ uriPath(localShardTmpdir) } && gtar xf $shardName)")
    log.info(s"untar'ing GenomicsDB shard with: $untarCommand")
    val rc = untarCommand !

    if (rc != 0)
      fatal(s"GenomicsDB shard untar failed with return code: $rc")

    assert(localShard.endsWith(".tar"))
    val workspace = uriPath(localShard.dropRight(4))

    val fastaFilename = metadata.fastaFilename

    log.info(s"workspace = $workspace")

    val gdbReader = new GenomicsDBFeatureReader(
      new File(workspace, DEFAULT_VIDMAP_FILE_NAME).getAbsolutePath,
      new File(workspace, DEFAULT_CALLSETMAP_FILE_NAME).getAbsolutePath,
      workspace,
      DEFAULT_ARRAY_NAME,
      fastaFilename,
      new File(workspace, DEFAULT_VCFHEADER_FILE_NAME).getAbsolutePath,
      new BCF2Codec())

    assert(gdbReader.getHeader.asInstanceOf[VCFHeader].getSampleNamesInOrder.asScala
      == metadata.sampleIds.toFastSeq)

    val reader = new HtsjdkRecordReader(metadata.callFields)

    val region = Region()
    val rvb = new RegionValueBuilder(region)
    val rv = RegionValue(region)

    val it: java.util.Iterator[VariantContext] = gdbReader.iterator
    it.asScala
      .map { vc =>
        region.clear()
        rvb.start(typ.rvRowType)
        reader.readRecord(vc, rvb, metadata.infoType, typ.entryType, dropSamples = false, metadata.canonicalFlags, metadata.infoFlagFieldNames)
        rv.setOffset(rvb.end())
        rv
      }
  }

  def formatHeaderSignature(
    lines: java.util.Collection[VCFFormatHeaderLine],
    callFields: Set[String],
    arrayElementsRequired: Boolean
  ): (TStruct, Int, VCFAttributes) = {
    val canonicalFields = Array(
      "GT" -> TCall(),
      "AD" -> TArray(TInt32(arrayElementsRequired)),
      "DP" -> TInt32(),
      "GQ" -> TInt32(),
      "PL" -> TArray(TInt32(arrayElementsRequired)))

    val (raw, attrs, _) = LoadVCF.headerSignature(lines, callFields, arrayElementsRequired)

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

  def apply(metadataFilename: String,
    vcfHeaderFilename: String,
    fastaFilename: String,
    callFields: Set[String],
    rg0: Option[ReferenceGenome],
    arrayElementsRequired: Boolean
  ): MatrixTable = {
    val hc = HailContext.get
    val hConf = hc.hadoopConf

    val rg = rg0.getOrElse(ReferenceGenome.defaultReference)

    val metadataPath = new hadoop.fs.Path(metadataFilename)
    val basePath = metadataPath.getParent
    val baseDirname = basePath.toString

    val fileMetadata =
      hConf
        .readFile(metadataFilename) { in =>
          implicit val formats = defaultJSONFormats
          val jv = parse(in)
          jv.extract[GenomicsDBFileMetadata]
        }

    val headerLines = LoadVCF.getHeaderLines(hConf, baseDirname + "/" + fileMetadata.vcfheader)

    val codec = new htsjdk.variant.vcf.VCFCodec()
    val header = codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
      .getHeaderValue
      .asInstanceOf[VCFHeader]

    val infoHeader = header.getInfoHeaderLines
    val (infoSignature, infoAttrs, infoFlagFieldNames) = LoadVCF.headerSignature(infoHeader)

    val formatHeader = header.getFormatHeaderLines
    val (genotypeSignature, canonicalFlags, formatAttrs) = formatHeaderSignature(formatHeader, callFields, arrayElementsRequired)

    val rowType = TStruct(
      "locus" -> TLocus.schemaFromRG(Some(rg)),
      "alleles" -> TArray(TString()),
      "rsid" -> TString(),
      "qual" -> TFloat64(),
      "filters" -> TSet(TString()),
      "info" -> infoSignature)

    val typ = MatrixType.fromParts(
      TStruct.empty(true),
      colType = TStruct("s" -> TString()),
      colKey = Array("s"),
      rowType = rowType,
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = genotypeSignature)

    val metadata = GenomicsDBMetadata(rg, fastaFilename, baseDirname, fileMetadata.sample_names, typ, infoSignature, callFields, canonicalFlags, infoFlagFieldNames,
      fileMetadata.shards
        .map { case JObject(fields) =>
          assert(fields.length == 1)
          fields.head match {
            case (interval, JString(path)) =>
              GenomicsDBShard(Parser.parseLocusInterval(interval, rg), path)
          }
        })

    val colValues = BroadcastIndexedSeq(
      fileMetadata.sample_names
        .map { Row(_) },
      TArray(typ.colType),
      hc.sc)

    new MatrixTable(hc, MatrixImportGenomicsDB(typ, metadata, colValues))
  }
}

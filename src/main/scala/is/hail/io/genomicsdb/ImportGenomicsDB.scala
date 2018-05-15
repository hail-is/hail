package is.hail.io.genomicsdb

import java.io.File
import java.util.Optional

import com.intel.genomicsdb.model.GenomicsDBExportConfiguration
import com.intel.genomicsdb.reader.GenomicsDBFeatureReader
import htsjdk.tribble.readers.PositionalBufferedStream
import htsjdk.variant.bcf2.BCF2Codec
import htsjdk.variant.variantcontext.{GenotypeLikelihoods, VariantContext}
import htsjdk.variant.vcf.{VCFFormatHeaderLine, VCFHeader}
import is.hail.utils._
import is.hail.HailContext
import is.hail.annotations.{BroadcastIndexedSeq, RegionValue}
import is.hail.expr.{MatrixImportGenomicsDB, Parser}
import is.hail.expr.types._
import is.hail.io.VCFAttributes
import is.hail.io.vcf.{BufferedLineIterator, HtsjdkRecordReader, LoadVCF}
import is.hail.rvd.RVDContext
import is.hail.utils.Interval
import is.hail.variant.{Locus, MatrixTable, ReferenceGenome}
import org.apache.hadoop
import org.apache.hadoop.fs.Path
import org.apache.spark.{ExposedMetrics, TaskContext}
import org.apache.spark.sql.Row
import org.json4s.JObject
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods.parse

import scala.collection.JavaConverters._
import scala.collection.mutable

case class GenomicsDBShard(
  interval: Interval,
  filename: String)

case class GenomicsDBFileMetadata(
  sample_names: Array[String],
  shards: Array[JObject],
  vcfheader: String)

case class GenomicsDBMetadata(
  rg: ReferenceGenome,
  baseDirname: String,
  sampleIds: Array[String],
  typ: MatrixType,
  infoType: TStruct,
  callFields: Set[String],
  tar: String,
  canonicalFlags: Int,
  infoFlagFieldNames: Set[String],
  shards: Array[GenomicsDBShard])

object ImportGenomicsDB {
  val DEFAULT_ARRAY_NAME: String = "genomicsdb_array"
  val DEFAULT_VIDMAP_FILE_NAME: String = "vidmap.json"
  val DEFAULT_CALLSETMAP_FILE_NAME: String = "callset.json"
  val DEFAULT_VCFHEADER_FILE_NAME: String = "vcfheader.vcf"

  def readShard(ctx: RVDContext, hadoopConf: hadoop.conf.Configuration, metadata: GenomicsDBMetadata, typ: MatrixType, i: Int): Iterator[RegionValue] = {
    val localShardTmpdir = hadoopConf.getTemporaryFile("file:///tmp", prefix = Some("genomicsdb-shard-"))
    hadoopConf.mkDir(localShardTmpdir)

    val tar = metadata.tar
    val cmd = Array(tar, "xf", "-", "-C", uriPath(localShardTmpdir))
    log.info(s"untar command: ${ cmd.mkString(" ") }")

    val pb = new ProcessBuilder(cmd.toList.asJava)
    val proc = pb.start()

    // unused
    proc.getInputStream.close()
    proc.getErrorStream.close()

    val shard = metadata.shards(i)
    val shardFilename = metadata.baseDirname + "/" + shard.filename

    hadoopConf.readFile(shardFilename) { shardIS =>
      val procOS = proc.getOutputStream

      val n = 64 * 1024
      val buf = new Array[Byte](n)
      var read: Int = 0
      do {
        read = shardIS.read(buf)
        if (read > 0)
          procOS.write(buf, 0, read)
      } while (read != -1)

      procOS.close()
    }

    val rc = proc.waitFor()

    if (rc != 0)
      fatal(s"shard $i untar failed with return code: $rc")

    val shardPath = new Path(shard.filename)
    val shardName = shardPath.getName
    assert(shardName.endsWith(".tar"))
    val workspaceDirName = shardName.dropRight(4)

    val workspace = uriPath(localShardTmpdir + "/" + workspaceDirName)
    log.info(s"shard $i workspace: $workspace")

    val fastaFilename = metadata.rg.localFastaFile

    val exportConf = GenomicsDBExportConfiguration.ExportConfiguration.newBuilder()
      .setWorkspace(workspace)
      .setReferenceGenome(fastaFilename)
      .setVidMappingFile(new File(workspace, DEFAULT_VIDMAP_FILE_NAME).getAbsolutePath)
      .setCallsetMappingFile(new File(workspace, DEFAULT_CALLSETMAP_FILE_NAME).getAbsolutePath)
      .setVcfHeaderFilename(new File(workspace, DEFAULT_VCFHEADER_FILE_NAME).getAbsolutePath)
      .setProduceGTField(true)
      .setProduceGTWithMinPLValueForSpanningDeletions(false)
      .setSitesOnlyQuery(false)
      .setMaxDiploidAltAllelesThatCanBeGenotyped(GenotypeLikelihoods.MAX_DIPLOID_ALT_ALLELES_THAT_CAN_BE_GENOTYPED)
      .setArrayName(DEFAULT_ARRAY_NAME)
      .build()

    val gdbReader = new GenomicsDBFeatureReader[VariantContext, PositionalBufferedStream](exportConf, new BCF2Codec(), Optional.empty[String]())

    assert(gdbReader.getHeader.asInstanceOf[VCFHeader].getSampleNamesInOrder.asScala
      == metadata.sampleIds.toFastSeq)

    val reader = new HtsjdkRecordReader(metadata.callFields)

    val region = ctx.region
    val rvb = ctx.rvb
    val rv = RegionValue(region)

    val context = TaskContext.get
    val inputMetrics = context.taskMetrics().inputMetrics

    val tlocus = TLocus(metadata.rg)
    val tlocusEOrd = tlocus.ordering

    val si = shard.interval
    val start = si.start.asInstanceOf[Locus]
    val end = si.end.asInstanceOf[Locus]
    val it: java.util.Iterator[VariantContext] = gdbReader.iterator
    it.asScala
      .flatMap { vc =>
        rvb.start(typ.rvRowType)
        reader.readRecord(vc, rvb, metadata.infoType, typ.entryType, dropSamples = false, metadata.canonicalFlags, metadata.infoFlagFieldNames)
        rv.setOffset(rvb.end())

        val locus = Locus(vc.getContig, vc.getStart)
        if (si.contains(tlocusEOrd, locus)) {
          ExposedMetrics.incrementRecord(inputMetrics)
          ExposedMetrics.incrementBytes(inputMetrics, 1)
          Some(rv)
        } else
          None
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
    callFields: Set[String],
    rg0: Option[ReferenceGenome],
    arrayElementsRequired: Boolean,
    tar: String
  ): MatrixTable = {
    val hc = HailContext.get
    val hConf = hc.hadoopConf

    val rg = rg0.getOrElse(ReferenceGenome.defaultReference)
    if (!rg.hasSequence)
      fatal(s"Reference genome '${ rg.name }' does not have sequence loaded.")

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

    val tlocus = TLocus.schemaFromRG(Some(rg))
    val rowType = TStruct(
      "locus" -> tlocus,
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

    val metadata = GenomicsDBMetadata(rg, baseDirname, fileMetadata.sample_names, typ, infoSignature, callFields, tar, canonicalFlags, infoFlagFieldNames,
      fileMetadata.shards
        .map { case JObject(fields) =>
          assert(fields.length == 1)
          fields.head match {
            case (interval, JString(path)) =>
              val si = Parser.parseLocusInterval(interval, rg)
              assert(si.includesStart && si.includesEnd)
              assert(si.start.asInstanceOf[Locus].contig == si.end.asInstanceOf[Locus].contig)
              GenomicsDBShard(si, path)
          }
        })

    // validate metadata
    val locusEOrd = tlocus.ordering
    var i = 0
    while (i < metadata.shards.length - 1) {
      val sEnd = metadata.shards(i).interval.end.asInstanceOf[Locus]
      val nextStart = metadata.shards(i + 1).interval.start.asInstanceOf[Locus]
      if (locusEOrd.gteq(sEnd, nextStart))
        fatal(s"shard intervals not sorted: shard $i end $sEnd >= shard ${ i + 1 } start $nextStart")
      i += 1
    }

    val colValues = BroadcastIndexedSeq(
      fileMetadata.sample_names.map(Row(_)),
      TArray(typ.colType),
      hc.sc)

    new MatrixTable(hc, MatrixImportGenomicsDB(typ, metadata, colValues))
  }
}

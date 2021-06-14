package is.hail.io.bgen

import is.hail.backend.BroadcastValue
import is.hail.backend.spark.SparkTaskContext
import is.hail.expr.ir.ExecuteContext
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.{RVD, RVDPartitioner, RVDType}
import is.hail.types.TableType
import is.hail.types.physical.{PCanonicalStruct, PStruct}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, TaskContext}

private case class IndexBgenPartition(
  path: String,
  compressed: Boolean,
  skipInvalidLoci: Boolean,
  contigRecoding: Map[String, String],
  startByteOffset: Long,
  endByteOffset: Long,
  partitionIndex: Int,
  fsBc: BroadcastValue[FS]
) extends BgenPartition {

  def index = partitionIndex
}

object IndexBgen {

  val bufferSpec: BufferSpec = LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024,
      LZ4HCBlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)))

  def apply(
    ctx: ExecuteContext,
    files: Array[String],
    indexFileMap: Map[String, String] = null,
    rg: Option[String] = None,
    contigRecoding: Map[String, String] = null,
    skipInvalidLoci: Boolean = false
  ) {
    val fs = ctx.fs
    val fsBc = fs.broadcast

    val statuses = LoadBgen.getAllFileStatuses(fs, files)
    val bgenFilePaths = statuses.map(_.getPath.toString)
    val indexFilePaths = LoadBgen.getIndexFileNames(fs, bgenFilePaths, indexFileMap)

    indexFilePaths.foreach { f =>
      assert(f.endsWith(".idx2"))
      if (fs.exists(f))
        fs.delete(f, recursive = true)
    }

    val recoding = Option(contigRecoding).getOrElse(Map.empty[String, String])
    val referenceGenome = rg.map(ReferenceGenome.getReference)
    referenceGenome.foreach(_.validateContigRemap(recoding))

    val headers = LoadBgen.getFileHeaders(fs, bgenFilePaths)
    LoadBgen.checkVersionTwo(headers)

    val annotationType = +PCanonicalStruct()

    val settings: BgenSettings = BgenSettings(
      0, // nSamples not used if there are no entries
      TableType(rowType = TStruct(
        "locus" -> TLocus.schemaFromRG(referenceGenome),
        "alleles" -> TArray(TString),
        "offset" -> TInt64,
        "file_idx" -> TInt32),
        key = Array("locus", "alleles"),
        globalType = TStruct.empty),
      referenceGenome.map(_.broadcast),
      annotationType.virtualType
    )

    val typ = RVDType(settings.rowPType, Array("file_idx", "locus", "alleles"))

    val partitions: Array[Partition] = headers.zipWithIndex.map { case (f, i) =>
      IndexBgenPartition(
        f.path,
        f.compressed,
        skipInvalidLoci,
        recoding,
        f.dataStart,
        f.fileByteSize,
        i,
        fsBc)
    }

    val rowType = typ.rowType
    val locusIdx = rowType.fieldIdx("locus")
    val allelesIdx = rowType.fieldIdx("alleles")
    val offsetIdx = rowType.fieldIdx("offset")
    val fileIdxIdx = rowType.fieldIdx("file_idx")
    val (keyType, _) = rowType.virtualType.select(Array("file_idx", "locus", "alleles"))
    val indexKeyType = rowType.selectFields(Array("locus", "alleles")).setRequired(false).asInstanceOf[PStruct]

    val attributes = Map("reference_genome" -> rg.orNull,
      "contig_recoding" -> recoding,
      "skip_invalid_loci" -> skipInvalidLoci)

    val rangeBounds = bgenFilePaths.zipWithIndex.map { case (_, i) => Interval(Row(i), Row(i), includesStart = true, includesEnd = true) }
    val partitioner = new RVDPartitioner(Array("file_idx"), keyType.asInstanceOf[TStruct], rangeBounds)
    val crvd = BgenRDD(ctx, partitions, settings, null).toCRDDPtr

    val makeIW = IndexWriter.builder(ctx, indexKeyType, annotationType, attributes = attributes)

    RVD.unkeyed(rowType, crvd)
      .repartition(ctx, partitioner, shuffle = true)
      .toRows
      .foreachPartition { it =>
        val partIdx = TaskContext.get.partitionId()
        val idxPath = indexFilePaths(partIdx)
        val htc = SparkTaskContext.get()

        htc.getRegionPool().scopedRegion { r =>
          using(makeIW(idxPath, r.pool)) { iw =>
            it.foreach { r =>
              assert(r.getInt(fileIdxIdx) == partIdx)
              iw.appendRow(Row(r(locusIdx), r(allelesIdx)), r.getLong(offsetIdx), Row())
            }
          }
        }
        info(s"Finished writing index file for ${ bgenFilePaths(partIdx) }")
      }
  }
}

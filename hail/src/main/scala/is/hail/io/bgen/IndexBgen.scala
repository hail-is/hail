package is.hail.io.bgen

import is.hail.HailContext
import is.hail.annotations.{SafeRow, UnsafeRow}
import is.hail.expr.types.TStruct
import is.hail.io.HadoopFSDataBinaryReader
import is.hail.io.index.{IndexReader, IndexWriter}
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVD}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row

private case class IndexBgenPartition(
  path: String,
  compressed: Boolean,
  skipInvalidLoci: Boolean,
  contigRecoding: Map[String, String],
  startByteOffset: Long,
  endByteOffset: Long,
  partitionIndex: Int,
  sHadoopConfBc: Broadcast[SerializableHadoopConfiguration]
) extends BgenPartition {

  def index = partitionIndex
}

object IndexBgen {
  def apply(
    hc: HailContext,
    files: Array[String],
    indexFileMap: Map[String, String] = null,
    rg: Option[String] = None,
    contigRecoding: Map[String, String] = null,
    skipInvalidLoci: Boolean = false) {
    val hConf = hc.hadoopConf

    val statuses = LoadBgen.getAllFileStatuses(hConf, files)
    val bgenFilePaths = statuses.map(_.getPath.toString)
    val indexFilePaths = LoadBgen.getIndexFileNames(hConf, files, indexFileMap)

    indexFilePaths.foreach { f =>
      assert(f.endsWith(".idx2"))
      if (hConf.exists(f))
        hConf.delete(f, recursive = true)
    }

    val recoding = Option(contigRecoding).getOrElse(Map.empty[String, String])
    val referenceGenome = rg.map(ReferenceGenome.getReference)
    referenceGenome.foreach(_.validateContigRemap(recoding))

    val headers = LoadBgen.getFileHeaders(hConf, bgenFilePaths)
    LoadBgen.checkVersionTwo(headers)

    val annotationType = +TStruct()

    val settings: BgenSettings = BgenSettings(
      0, // nSamples not used if there are no entries
      NoEntries,
      RowFields(false, false, true),
      referenceGenome,
      annotationType
    )

    val typ = new OrderedRVDType(Array("locus", "alleles"), settings.typ)

    val sHadoopConfBc = hc.sc.broadcast(new SerializableHadoopConfiguration(hConf))

    val rvds = headers.map { f =>
      val partition = IndexBgenPartition(
        f.path,
        f.compressed,
        skipInvalidLoci,
        recoding,
        f.dataStart,
        f.fileByteSize,
        0,
        sHadoopConfBc)

      val crvd = BgenRDD(hc.sc, Array(partition), settings, null)
      OrderedRVD.coerce(typ, crvd)
    }

    val rowType = typ.rowType
    val offsetIdx = rowType.fieldIdx("offset")
    val (keyType, kf) = rowType.select(Array("locus", "alleles"))

    val attributes = Map("reference_genome" -> rg.orNull,
      "contig_recoding" -> recoding,
      "skip_invalid_loci" -> skipInvalidLoci)

    val unionRVD = RVD.union(rvds)
    assert(unionRVD.getNumPartitions == files.length)

    unionRVD.toRows.foreachPartition({ it =>
      val partIdx = TaskContext.get.partitionId()
      using(new IndexWriter(sHadoopConfBc.value.value, indexFilePaths(partIdx), keyType, annotationType, attributes = attributes)) { iw =>
        it.foreach { row =>
          iw += (row.deleteField(offsetIdx), row.getLong(offsetIdx), Row())
        }
      }
    })
  }
}

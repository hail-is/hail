package is.hail.io.bgen

import java.util.concurrent.Executors

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

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

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
    val indexFilePaths = LoadBgen.getIndexFileNames(hConf, bgenFilePaths, indexFileMap)

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

    val rowType = typ.rowType
    val offsetIdx = rowType.fieldIdx("offset")
    val (keyType, _) = rowType.select(Array("locus", "alleles"))

    val attributes = Map("reference_genome" -> rg.orNull,
      "contig_recoding" -> recoding,
      "skip_invalid_loci" -> skipInvalidLoci)

    implicit val ec = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(headers.length))

    val rvdFutures = headers.zipWithIndex.map { case (f, i) =>
      Future({
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
        assert(crvd.getNumPartitions == 1)

        OrderedRVD.coerce(typ, crvd).toRows.foreachPartition({ it =>
          using(new IndexWriter(sHadoopConfBc.value.value, indexFilePaths(i), keyType, annotationType, attributes = attributes)) { iw =>
            it.foreach { row =>
              iw += (row.deleteField(offsetIdx), row.getLong(offsetIdx), Row())
            }
          }
        })
        info(s"Finished indexing file '${ f.path }'")
      })
    }

    Await.result(Future.sequence(rvdFutures.toFastIndexedSeq), Duration.Inf)
  }
}

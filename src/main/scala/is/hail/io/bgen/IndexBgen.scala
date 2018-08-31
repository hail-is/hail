package is.hail.io.bgen

import is.hail.HailContext
import is.hail.annotations.UnsafeRow
import is.hail.expr.types.TStruct
import is.hail.io.HadoopFSDataBinaryReader
import is.hail.io.index.IndexWriter
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVD}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.hadoop.fs.Path
import org.apache.spark.Partition
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row

private case class IndexBgenPartition(
  path: String,
  compressed: Boolean,
  startByteOffset: Long,
  endByteOffset: Long,
  partitionIndex: Int,
  sHadoopConfBc: Broadcast[SerializableHadoopConfiguration]
) extends BgenPartition {

  def index = partitionIndex

  def makeInputStream = {
    val hadoopPath = new Path(path)
    val fs = hadoopPath.getFileSystem(sHadoopConfBc.value.value)
    val bfis = new HadoopFSDataBinaryReader(fs.open(hadoopPath))
    bfis.seek(startByteOffset)
    bfis
  }

  def advance(bfis: HadoopFSDataBinaryReader) { }

  def hasNext(bfis: HadoopFSDataBinaryReader): Boolean =
    bfis.getPosition < endByteOffset
}

object IndexBgen {
  def makePartitions(files: Array[BgenHeader]): Array[Partition] = {
    files.zipWithIndex.map { case (f, i) =>
      IndexBgenPartition(f.path, f.compressed, f.dataStart, f.fileByteSize, i, ???)
    }
  }

  def apply(
    hc: HailContext,
    files: Array[String],
    rg: Option[String] = None,
    contigRecoding: Map[String, String] = null,
    skipInvalidLoci: Boolean = false) {
    val hConf = hc.hadoopConf

    val referenceGenome = rg.map(ReferenceGenome.getReference)
    referenceGenome.foreach(_.validateContigRemap(contigRecoding))

    val headers = LoadBgen.getFileHeaders(hConf, files)
    LoadBgen.checkVersionTwo(headers)

    val settings: BgenSettings = BgenSettings(
      0, // not used if there are no entries
      NoEntries,
      RowFields(false, false, false),
      referenceGenome,
      contigRecoding,
      skipInvalidLoci
    )

    val typ: OrderedRVDType = settings.matrixType.orvdType

    val sHadoopConfBc = hc.sc.broadcast(new SerializableHadoopConfiguration(hConf))

    val rvds = headers.map { f =>
      val partition = IndexBgenPartition(
        f.path,
        f.compressed,
        f.dataStart,
        f.fileByteSize,
        0,
        sHadoopConfBc)

      val crvd = BgenRDD(hc.sc, Array(partition), settings)
      OrderedRVD.coerce(typ, crvd)
    }

    val rowType = typ.rowType
    val offsetIdx = rowType.fieldIdx("offset")
    val (keyType, kf) = rowType.select(Array("locus", "alleles"))

    val attributes = Map("reference_genome" -> rg.orNull,
      "contig_recoding" -> Option(contigRecoding).getOrElse(Map.empty[String, String]),
      "skip_invalid_loci" -> skipInvalidLoci)

    val unionRVD = RVD.union(rvds)
    assert(unionRVD.getNumPartitions == files.length)

    val serializableHadoopConf = new SerializableHadoopConfiguration(hConf)
    unionRVD.boundary.mapPartitionsWithIndex({ (i, it) =>
      val iw = new IndexWriter(serializableHadoopConf, files(i) + ".idx2", keyType, +TStruct(), attributes = attributes)
      it.foreach { rv =>
        val r = UnsafeRow.readBaseStruct(rowType, rv.region, rv.offset)
        iw += (kf(r), r.getLong(offsetIdx), Row())
      }
      iw.close()
      Iterator.single(1)
    }).collect()
  }
}

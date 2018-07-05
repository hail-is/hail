package is.hail.io.bgen

import is.hail.io.{ HadoopFSDataBinaryReader, OnDiskBTreeIndexToValue }
import org.apache.hadoop.fs.Path
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{ Partition, SparkContext }
import is.hail.utils._
import scala.annotation.switch

trait BgenPartition extends Partition {
  def path: String

  def compressed: Boolean

  def makeInputStream: HadoopFSDataBinaryReader

  // advances the reader to the next variant position and returns the index of
  // said variant
  def advance(bfis: HadoopFSDataBinaryReader): Long

  def hasNext(bfis: HadoopFSDataBinaryReader): Boolean
}

object BgenRDDPartitions extends Logging {
  private case class BgenFileMetadata (
    file: String,
    byteFileSize: Long,
    dataByteOffset: Long,
    nVariants: Int,
    compressed: Boolean
  )

  def apply(
    sc: SparkContext,
    files: Seq[String],
    minPartitions: Int,
    includedVariantsPerFile: Map[String, Seq[Int]],
    settings: BgenSettings
  ): Array[Partition] = {
    val hConf = sc.hadoopConfiguration
    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))
    val fileMetadata = files.map { file =>
      val state = LoadBgen.readState(hConf, file)
      val keptVariants = includedVariantsPerFile.get(file)
        .map(_.length)
        .getOrElse(state.nVariants)
      BgenFileMetadata(file, hConf.getFileSize(file), state.dataStart, keptVariants, state.compressed)
    }
    val nonEmptyFileMetadatas = fileMetadata.filter(_.nVariants > 0)
    if (nonEmptyFileMetadatas.isEmpty) {
      Array.empty
    } else {
      val metadata = nonEmptyFileMetadatas(0)
      val recordByteSizeEstimate =
        (metadata.byteFileSize - metadata.dataByteOffset) / metadata.nVariants
      val minBlockSize =
        hConf.getInt("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", 0)
      var minRecordsPerPartition =
        (minBlockSize / recordByteSizeEstimate).ceil.toInt
      var maxRecordsPerPartition =
        (settings.nVariants / minPartitions).floor.toInt
      if (maxRecordsPerPartition == 0) {
        log.warn(
          s"""only found ${settings.nVariants} but min_partitions is
             |$minPartitions. Actual number of partitions will be
             |fewer.""".stripMargin)
        maxRecordsPerPartition = 1
      }
      if (maxRecordsPerPartition < minRecordsPerPartition) {
        log.warn(
          s"""cannot satisfy requested min_partitions ($minPartitions) and
             |min_block_size ($minBlockSize) for BGEN files $files from which
             |we are loading ${settings.nVariants} variants, with an average
             |size of $recordByteSizeEstimate bytes.""".stripMargin)
        maxRecordsPerPartition = minRecordsPerPartition
      }
      // FIXME: divide balls as evenly as possible into some number of buckets
      // where each bucket has no more than maxRecordsPerPartition and no less
      // than minRecordsPerPartition
      val partitions = new ArrayBuilder[Partition]()
      var partitionIndex = 0
      var metadataIndex = 0
      while (metadataIndex < nonEmptyFileMetadatas.length) {
        val metadata = nonEmptyFileMetadatas(metadataIndex)
        using(new OnDiskBTreeIndexToValue(metadata.file + ".idx", hConf)) { index =>
          val nPartitions = (metadata.nVariants / maxRecordsPerPartition).ceil.toInt
          includedVariantsPerFile.get(metadata.file) match {
            case None =>
              val startOffsets =
                index.positionOfVariants(
                  Array.tabulate(nPartitions)(i => i * maxRecordsPerPartition))
              var i = 0
              while (i < nPartitions - 1) {
                partitions += BgenPartitionWithoutFilter(
                  metadata.file,
                  metadata.compressed,
                  i * maxRecordsPerPartition,
                  startOffsets(i),
                  startOffsets(i + 1),
                  partitionIndex,
                  sHadoopConfBc
                )
                partitionIndex += 1
                i += 1
              }
              partitions += BgenPartitionWithoutFilter(
                metadata.file,
                metadata.compressed,
                i * maxRecordsPerPartition,
                startOffsets(i),
                metadata.byteFileSize,
                partitionIndex,
                sHadoopConfBc
              )
              partitionIndex += 1
            case Some(variantIndices) =>
              val startOffsets =
                index.positionOfVariants(variantIndices.toArray)
              val parts = new Array[Partition](nPartitions)
              var i = 0
              while (i < nPartitions - 1) {
                val left = i * maxRecordsPerPartition
                val rightExclusive = (i + 1) * maxRecordsPerPartition
                partitions += BgenPartitionWithFilter(
                  metadata.file,
                  metadata.compressed,
                  partitionIndex,
                  startOffsets.slice(left, rightExclusive),
                  variantIndices.slice(left, rightExclusive).toArray,
                  sHadoopConfBc
                )
                i += 1
                partitionIndex += 1
              }
              val left = i * maxRecordsPerPartition
              partitions += BgenPartitionWithFilter(
                metadata.file,
                metadata.compressed,
                partitionIndex,
                startOffsets.slice(left, startOffsets.length),
                variantIndices.slice(left, variantIndices.length).toArray,
                sHadoopConfBc
              )
              partitionIndex += 1
          }
        }
        metadataIndex += 1
      }
      partitions.result()
    }
  }

  private case class BgenPartitionWithoutFilter (
    path: String,
    compressed: Boolean,
    firstRecordIndex: Long,
    startByteOffset: Long,
    endByteOffset: Long,
    partitionIndex: Int,
    sHadoopConfBc: Broadcast[SerializableHadoopConfiguration]
  ) extends BgenPartition {
    private[this] var records = firstRecordIndex - 1

    def index = partitionIndex

    def makeInputStream = {
      val hadoopPath = new Path(path)
      val fs = hadoopPath.getFileSystem(sHadoopConfBc.value.value)
      val bfis = new HadoopFSDataBinaryReader(fs.open(hadoopPath))
      bfis.seek(startByteOffset)
      bfis
    }

    def advance(bfis: HadoopFSDataBinaryReader): Long = {
      records += 1
      records
    }

    def hasNext(bfis: HadoopFSDataBinaryReader): Boolean =
      bfis.getPosition < endByteOffset
  }

  private case class BgenPartitionWithFilter (
    path: String,
    compressed: Boolean,
    partitionIndex: Int,
    keptVariantOffsets: Array[Long],
    keptVariantIndices: Array[Int],
    sHadoopConfBc: Broadcast[SerializableHadoopConfiguration]
  ) extends BgenPartition {
    private[this] var keptVariantIndex = -1
    assert(keptVariantOffsets != null)
    assert(keptVariantIndices != null)
    assert(keptVariantOffsets.length == keptVariantIndices.length)

    def index = partitionIndex

    def makeInputStream = {
      val hadoopPath = new Path(path)
      val fs = hadoopPath.getFileSystem(sHadoopConfBc.value.value)
      new HadoopFSDataBinaryReader(fs.open(hadoopPath))
    }

    def advance(bfis: HadoopFSDataBinaryReader): Long = {
      keptVariantIndex += 1
      bfis.seek(keptVariantOffsets(keptVariantIndex))
      keptVariantIndices(keptVariantIndex)
    }

    def hasNext(bfis: HadoopFSDataBinaryReader): Boolean =
      keptVariantIndex < keptVariantIndices.length - 1
  }
}

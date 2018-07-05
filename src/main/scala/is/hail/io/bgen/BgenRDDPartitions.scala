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
    files: Seq[BgenHeader],
    minPartitions: Int,
    includedVariantsPerFile: Map[String, Seq[Int]],
    settings: BgenSettings
  ): Array[Partition] = {
    val hConf = sc.hadoopConfiguration
    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))
    val filesWithVariantFilters = files.map { header =>
      val nKeptVariants = includedVariantsPerFile.get(header.path)
        .map(_.length)
        .getOrElse(header.nVariants)
      header.copy(nVariants = nKeptVariants)
    }
    val nonEmptyFilesAfterFilter = filesWithVariantFilters.filter(_.nVariants > 0)
    if (nonEmptyFilesAfterFilter.isEmpty) {
      Array.empty
    } else {
      val metadata = nonEmptyFilesAfterFilter(0)
      val recordByteSizeEstimate =
        (metadata.fileByteSize - metadata.dataStart) / metadata.nVariants
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
             |min_block_size ($minBlockSize) for BGEN files ${files.map(_.path)} from which
             |we are loading ${settings.nVariants} variants, with an average
             |size of $recordByteSizeEstimate bytes.""".stripMargin)
        maxRecordsPerPartition = minRecordsPerPartition
      }
      // FIXME: divide balls as evenly as possible into some number of buckets
      // where each bucket has no more than maxRecordsPerPartition and no less
      // than minRecordsPerPartition
      val partitions = new ArrayBuilder[Partition]()
      var partitionIndex = 0
      var fileIndex = 0
      while (fileIndex < nonEmptyFilesAfterFilter.length) {
        val file = nonEmptyFilesAfterFilter(fileIndex)
        using(new OnDiskBTreeIndexToValue(file.path + ".idx", hConf)) { index =>
          val nPartitions = (file.nVariants / maxRecordsPerPartition).ceil.toInt
          includedVariantsPerFile.get(file.path) match {
            case None =>
              val startOffsets =
                index.positionOfVariants(
                  Array.tabulate(nPartitions)(i => i * maxRecordsPerPartition))
              var i = 0
              while (i < nPartitions - 1) {
                partitions += BgenPartitionWithoutFilter(
                  file.path,
                  file.compressed,
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
                file.path,
                file.compressed,
                i * maxRecordsPerPartition,
                startOffsets(i),
                file.fileByteSize,
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
                  file.path,
                  file.compressed,
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
                file.path,
                file.compressed,
                partitionIndex,
                startOffsets.slice(left, startOffsets.length),
                variantIndices.slice(left, variantIndices.length).toArray,
                sHadoopConfBc
              )
              partitionIndex += 1
          }
        }
        fileIndex += 1
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

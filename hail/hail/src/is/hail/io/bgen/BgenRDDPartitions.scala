package is.hail.io.bgen

import is.hail.backend.ExecuteContext
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat._

case class FilePartitionInfo(
  metadata: BgenFileMetadata,
  intervals: Array[Interval],
  partStarts: Array[Long],
  partN: Array[Long],
)

object BgenRDDPartitions extends Logging {
  def checkFilesDisjoint(ctx: ExecuteContext, fileMetadata: Seq[BgenFileMetadata], keyType: Type)
    : Array[Interval] = {
    assert(fileMetadata.nonEmpty)
    val pord = keyType.ordering(ctx.stateManager)
    val bounds = fileMetadata.map(md => (md.path, md.rangeBounds))

    val overlappingBounds = ArraySeq.newBuilder[(String, Interval, String, Interval)]
    var i = 0
    while (i < bounds.length) {
      var j = 0
      while (j < i) {
        val b1 = bounds(i)
        val b2 = bounds(j)
        if (!b1._2.isDisjointFrom(pord, b2._2))
          overlappingBounds += ((b1._1, b1._2, b2._1, b2._2))
        j += 1
      }
      i += 1
    }

    overlappingBounds.result() match {
      case Seq() =>
      case overlappingBounds => fatal(
          s"""Each BGEN file must contain a region of the genome disjoint from other files. Found the following overlapping files:
             |  ${overlappingBounds.map { case (f1, i1, f2, i2) =>
              s"file1: $f1\trangeBounds1: $i1\tfile2: $f2\trangeBounds2: $i2"
            }.mkString("\n  ")})""".stripMargin
        )
    }

    bounds.map(_._2).toArray
  }

  def apply(
    ctx: ExecuteContext,
    rg: Option[String],
    files: IndexedSeq[BgenFileMetadata],
    blockSizeInMB: Option[Int],
    nPartitions: Option[Int],
    keyType: Type,
  ): IndexedSeq[FilePartitionInfo] = {
    val fileRangeBounds = checkFilesDisjoint(ctx, files, keyType)
    val intervalOrdering = TInterval(keyType).ordering(ctx.stateManager)

    val sortedFiles = files.zip(fileRangeBounds)
      .sortWith { case ((_, i1), (_, i2)) => intervalOrdering.lt(i1, i2) }
      .map(_._1)

    val totalSize = sortedFiles.map(_.header.fileByteSize).sum

    val fileNPartitions = (blockSizeInMB, nPartitions) match {
      case (Some(blockSizeInMB), _) =>
        val blockSizeInB = blockSizeInMB * 1024 * 1024
        sortedFiles.map { md =>
          val size = md.header.fileByteSize
          ((size + blockSizeInB - 1) / blockSizeInB).toInt
        }
      case (_, Some(nParts)) =>
        sortedFiles.map { md =>
          val size = md.header.fileByteSize
          ((size * nParts + totalSize - 1) / totalSize).toInt
        }
      case (None, None) => fatal(s"Must specify either of 'blockSizeInMB' or 'nPartitions'.")
    }

    val nonEmptyFilesAfterFilter = sortedFiles.filter(_.nVariants > 0)

    val indexSpec = BgenSettings.getIndexSpec(files.head.indexVersion, rg)
    val getKeysFromFile = StagedBGENReader.queryIndexByPosition(ctx, indexSpec)

    nonEmptyFilesAfterFilter.zipWithIndex.map { case (file, fileIndex) =>
      val nPartitions = math.min(fileNPartitions(fileIndex).toLong, file.nVariants).toInt
      val partNVariants: Array[Long] = partition(file.nVariants, nPartitions)
      val partFirstVariantIndex = partNVariants.scan(0L)(_ + _).init
      val partLastVariantIndex = partFirstVariantIndex.lazyZip(partNVariants).map { (idx, n) =>
        idx + n
      }

      val allPositions = partFirstVariantIndex ++ partLastVariantIndex.map(_ - 1L)
      val keys = getKeysFromFile(file.indexPath, allPositions)
      val rangeBounds = (0 until nPartitions).map { i =>
        Interval(
          keys(i),
          keys(i + nPartitions),
          true,
          true,
        ) // this must be true -- otherwise boundaries with duplicates will have the wrong range bounds
      }.toArray

      FilePartitionInfo(file, rangeBounds, partFirstVariantIndex, partNVariants)
    }
  }
}

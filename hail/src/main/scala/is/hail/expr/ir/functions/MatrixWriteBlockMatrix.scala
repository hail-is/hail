package is.hail.expr.ir.functions

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.MatrixValue
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, GridPartitioner, WriteBlocksRDD}
import is.hail.utils._

import java.io.DataOutputStream

import org.json4s.jackson

object MatrixWriteBlockMatrix {
  def apply(
    ctx: ExecuteContext,
    mv: MatrixValue,
    entryField: String,
    path: String,
    overwrite: Boolean,
    blockSize: Int,
  ): Unit = {
    val rvd = mv.rvd

    // FIXME
    val partitionCounts = rvd.countPerPartition()

    val fs = ctx.fs

    val partStarts = partitionCounts.scanLeft(0L)(_ + _)
    assert(partStarts.length == rvd.getNumPartitions + 1)

    val nRows = partStarts.last
    val localNCols = mv.nCols

    if (overwrite)
      fs.delete(path, recursive = true)
    else if (fs.exists(path))
      fatal(s"file already exists: $path")

    fs.mkDir(path)

    // write blocks
    fs.mkDir(path + "/parts")
    val gp = GridPartitioner(blockSize, nRows, localNCols)
    val blockPartFiles =
      new WriteBlocksRDD(fs.broadcast, ctx.localTmpdir, path, rvd, partStarts, entryField, gp)
        .collect()

    val blockCount = blockPartFiles.length
    val partFiles = new Array[String](blockCount)
    blockPartFiles.foreach { case (i, f) => partFiles(i) = f }

    // write metadata
    using(new DataOutputStream(fs.create(path + BlockMatrix.metadataRelativePath))) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, nRows, localNCols, gp.partitionIndexToBlockIndex, partFiles),
        os,
      )
    }

    assert(blockCount == gp.numPartitions)
    info(s"Wrote all $blockCount blocks of $nRows x $localNCols matrix with block size $blockSize.")

    using(fs.create(path + "/_SUCCESS"))(out => ())
  }
}

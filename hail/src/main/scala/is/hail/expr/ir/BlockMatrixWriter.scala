package is.hail.expr.ir

import is.hail.HailContext
import is.hail.io.fs.FS
import is.hail.linalg.BlockMatrix
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

object BlockMatrixWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[BlockMatrixNativeWriter], classOf[BlockMatrixBinaryWriter], classOf[BlockMatrixRectanglesWriter],
        classOf[BlockMatrixBinaryMultiWriter], classOf[BlockMatrixTextMultiWriter],
        classOf[BlockMatrixPersistWriter], classOf[BlockMatrixNativeMultiWriter]))
    override val typeHintFieldName: String = "name"
  }
}


abstract class BlockMatrixWriter {
  def pathOpt: Option[String]
  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit
}

case class BlockMatrixNativeWriter(
  path: String,
  overwrite: Boolean,
  forceRowMajor: Boolean,
  stageLocally: Boolean) extends BlockMatrixWriter {
  def pathOpt: Option[String] = Some(path)

  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit = bm.write(ctx, path, overwrite, forceRowMajor, stageLocally)
}

case class BlockMatrixBinaryWriter(path: String) extends BlockMatrixWriter {
  def pathOpt: Option[String] = Some(path)
  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit = {
    RichDenseMatrixDouble.exportToDoubles(ctx.fs, path, bm.toBreezeMatrix(), forceRowMajor = true)
  }
}

case class BlockMatrixPersistWriter(id: String, storageLevel: String) extends BlockMatrixWriter {
  def pathOpt: Option[String] = None
  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit =
    HailContext.sparkBackend("BlockMatrixPersistWriter").bmCache.persistBlockMatrix(id, bm, storageLevel)
}

case class BlockMatrixRectanglesWriter(
  path: String,
  rectangles: Array[Array[Long]],
  delimiter: String,
  binary: Boolean) extends BlockMatrixWriter {

  def pathOpt: Option[String] = Some(path)

  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit = {
    bm.exportRectangles(ctx, path, rectangles, delimiter, binary)
  }
}

abstract class BlockMatrixMultiWriter {
  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit
}

case class BlockMatrixBinaryMultiWriter(
  prefix: String,
  overwrite: Boolean) extends BlockMatrixMultiWriter {

  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit =
    BlockMatrix.binaryWriteBlockMatrices(ctx.fs, bms, prefix, overwrite)
}

case class BlockMatrixTextMultiWriter(
  prefix: String,
  overwrite: Boolean,
  delimiter: String,
  header: Option[String],
  addIndex: Boolean,
  compression: Option[String],
  customFilenames: Option[Array[String]]) extends BlockMatrixMultiWriter {

  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit =
    BlockMatrix.exportBlockMatrices(ctx.fs, bms, prefix, overwrite, delimiter, header, addIndex, compression, customFilenames)
}

case class BlockMatrixNativeMultiWriter(
  prefix: String,
  overwrite: Boolean,
  forceRowMajor: Boolean) extends BlockMatrixMultiWriter {

  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit = {
    BlockMatrix.writeBlockMatrices(ctx, bms, prefix, overwrite, forceRowMajor)
  }
}

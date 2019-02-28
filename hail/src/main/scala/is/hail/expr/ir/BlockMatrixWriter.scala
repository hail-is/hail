package is.hail.expr.ir

import is.hail.HailContext
import is.hail.linalg.BlockMatrix
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

object BlockMatrixWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[BlockMatrixNativeWriter], classOf[BlockMatrixBinaryWriter]))
    override val typeHintFieldName: String = "name"
  }
}


abstract class BlockMatrixWriter {
  def apply(hc: HailContext, bm: BlockMatrix): Unit
}

case class BlockMatrixNativeWriter(
  path: String,
  overwrite: Boolean,
  forceRowMajor: Boolean,
  stageLocally: Boolean) extends BlockMatrixWriter {

  def apply(hc: HailContext, bm: BlockMatrix): Unit = bm.write(path, overwrite, forceRowMajor, stageLocally)
}

case class BlockMatrixBinaryWriter(path: String) extends BlockMatrixWriter {
  def apply(hc: HailContext, bm: BlockMatrix): Unit = {
    RichDenseMatrixDouble.exportToDoubles(hc, path, bm.toBreezeMatrix(), forceRowMajor = true)
  }
}
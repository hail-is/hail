package is.hail.expr.ir

import is.hail.expr.types.virtual.Type
import is.hail.io._
import is.hail.io.gen.{ExportBGEN, ExportGen}
import is.hail.io.plink.ExportPlink
import is.hail.io.vcf.ExportVCF
import is.hail.utils.ExportType
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

object MatrixWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[MatrixNativeWriter], classOf[MatrixVCFWriter], classOf[MatrixGENWriter],
        classOf[MatrixBGENWriter], classOf[MatrixPLINKWriter], classOf[WrappedMatrixWriter]))
    override val typeHintFieldName = "name"
  }
}

case class WrappedMatrixWriter(writer: MatrixWriter,
  colsFieldName: String,
  entriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableWriter {
  def apply(tv: TableValue): Unit = writer(tv.toMatrixValue(colKey, colsFieldName, entriesFieldName))
}

abstract class MatrixWriter {
  def apply(mv: MatrixValue): Unit
}

case class MatrixNativeWriter(
  path: String,
  overwrite: Boolean = false,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null,
  partitions: String = null,
  partitionsTypeStr: String = null
) extends MatrixWriter {
  def apply(mv: MatrixValue): Unit = mv.write(path, overwrite, stageLocally, codecSpecJSONStr, partitions, partitionsTypeStr)
}

case class MatrixVCFWriter(
  path: String,
  append: Option[String] = None,
  exportType: Int = ExportType.CONCATENATED,
  metadata: Option[VCFMetadata] = None
) extends MatrixWriter {
  def apply(mv: MatrixValue): Unit = ExportVCF(mv, path, append, exportType, metadata)
}

case class MatrixGENWriter(
  path: String,
  precision: Int = 4
) extends MatrixWriter {
  def apply(mv: MatrixValue): Unit = ExportGen(mv, path, precision)
}

case class MatrixBGENWriter(
  path: String,
  exportType: Int
) extends MatrixWriter {
  def apply(mv: MatrixValue): Unit = ExportBGEN(mv, path, exportType)
}

case class MatrixPLINKWriter(
  path: String
) extends MatrixWriter {
  def apply(mv: MatrixValue): Unit = ExportPlink(mv, path)
}

object MatrixNativeMultiWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(classOf[MatrixNativeMultiWriter]))
    override val typeHintFieldName = "name"
  }
}

case class MatrixNativeMultiWriter(
  prefix: String,
  overwrite: Boolean = false,
  stageLocally: Boolean = false
) {
  def apply(mvs: IndexedSeq[MatrixValue]): Unit = MatrixValue.writeMultiple(mvs, prefix, overwrite, stageLocally)
}

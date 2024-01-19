package is.hail.expr

import is.hail.expr.ir.{
  BaseIR, BlockMatrixRead, BlockMatrixWrite, MatrixRead, MatrixWrite, TableRead, TableWrite,
}
import is.hail.utils._

case class ValidateState(writeFilePaths: Set[String])

object Validate {

  def apply(ir: BaseIR): Unit = validate(ir, ValidateState(Set()))

  def fileReadWriteError(path: String): Unit =
    fatal(s"path '$path' is both an input and output source in a single query. " +
      s"Write, export, or checkpoint to a different path!")

  private def validate(ir: BaseIR, state: ValidateState): Unit = {
    ir match {
      case tr: TableRead => tr.tr.pathsUsed.foreach { path =>
          if (state.writeFilePaths.contains(path))
            fileReadWriteError(path)
        }
      case mr: MatrixRead => mr.reader.pathsUsed.foreach { path =>
          if (state.writeFilePaths.contains(path))
            fileReadWriteError(path)
        }
      case _: BlockMatrixRead =>
      case tw: TableWrite =>
        val newState = state.copy(writeFilePaths = state.writeFilePaths + tw.writer.path)
        validate(tw.child, newState)
      case mw: MatrixWrite =>
        val newState = state.copy(writeFilePaths = state.writeFilePaths + mw.writer.path)
        validate(mw.child, newState)
      case bmw: BlockMatrixWrite =>
        val newState = bmw.writer.pathOpt match {
          case Some(path) => state.copy(writeFilePaths = state.writeFilePaths + path)
          case None => state
        }
        validate(bmw.child, newState)
      case _ => ir.children.foreach(validate(_, state))
    }
  }

}

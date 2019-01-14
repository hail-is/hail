package is.hail.expr.types

case class BlockMatrixType(nRows: Long, nCols: Long, blockSize: Int) extends BaseType {

  override def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    //TODO Used for Pretty printing of the BlockMatrix if things go wrong
  }
}

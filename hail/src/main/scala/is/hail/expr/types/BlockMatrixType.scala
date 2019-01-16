package is.hail.expr.types

case class BlockMatrixType(nRows: Long, nCols: Long, blockSize: Int) extends BaseType {

  override def pretty(sb: StringBuilder, indent0: Int, compact: Boolean): Unit = {
    var indent = indent0

    val space: String = if (compact) "" else " "

    def newline() {
      if (!compact) {
        sb += '\n'
        sb.append(" " * indent)
      }
    }

    sb.append(s"BlockMatrix$space{")
    indent += 4
    newline()

    sb.append(s"nRows:$space")
    sb.append(nRows)
    sb += ','
    newline()

    sb.append(s"nCols:$space")
    sb.append(nCols)
    sb += ','
    newline()

    sb.append(s"blockSize:$space")
    sb.append(blockSize)

    indent -= 4
    newline()
    sb += '}'
  }
}

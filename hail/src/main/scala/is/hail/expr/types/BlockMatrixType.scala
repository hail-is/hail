package is.hail.expr.types

import is.hail.utils._

case class BlockMatrixType(shape: IndexedSeq[Long], blockSize: Int, dimsPartitioned: IndexedSeq[Boolean]) extends BaseType {

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

    sb.append(s"shape:$space[")
    shape.foreachBetween(dimSize => sb.append(dimSize))(sb.append(s",$space"))
    sb += ']'
    sb += ','
    newline()

    sb.append(s"blockSize:$space[")
    sb.append(blockSize)
    sb += ','
    newline()

    sb.append(s"dimsPartitioned:$space[")
    dimsPartitioned.foreachBetween(dim => sb.append(dim))(sb.append(s",$space"))
    sb += ']'

    indent -= 4
    newline()
    sb += '}'
  }
}

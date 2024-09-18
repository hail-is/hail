package is.hail.types

package virtual {
  sealed abstract class Kind[T <: VType] extends Product with Serializable

  object Kinds {
    case object Value extends Kind[Type]
    case object Table extends Kind[TableType]
    case object Matrix extends Kind[MatrixType]
    case object BlockMatrix extends Kind[BlockMatrixType]
  }
}

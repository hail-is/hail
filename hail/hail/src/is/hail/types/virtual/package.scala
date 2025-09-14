package is.hail.types

package virtual {
  sealed abstract class Kind extends Product with Serializable

  object Kinds {
    case object Value extends Kind
    case object Table extends Kind
    case object Matrix extends Kind
    case object BlockMatrix extends Kind
  }
}

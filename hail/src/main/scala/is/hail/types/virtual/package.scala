package is.hail.types

package object virtual {

  object Primitive {
    def unapply(t: Type): Option[Type] =
      Some(t).filter(_.isPrimitive)
  }

}

package is.hail.expr.typ

import is.hail.annotations._

import scala.reflect.ClassTag

final case class TVariable(name: String, var t: Type = null) extends Type {
  override val required = true

  override def _toString: String = s"?$name"

  override def isRealizable = false

  def _typeCheck(a: Any): Boolean =
    throw new RuntimeException("TVariable is not realizable")

  override def unify(concrete: Type): Boolean = {
    if (t == null) {
      if (concrete.isRealizable) {
        t = concrete
        true
      } else
        false
    } else
      t.isOfType(concrete)
  }

  override def isBound: Boolean = t != null

  override def clear() {
    t = null
  }

  override def subst(): Type = {
    assert(t != null)
    t
  }

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TVariable is not realizable")

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    throw new RuntimeException("TVariable is not realizable")
}

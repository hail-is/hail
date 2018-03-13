package is.hail.expr.types

import is.hail.annotations._

import scala.reflect.ClassTag

final case class TVariable(name: String, cond: (Type) => Boolean = { _ => true }, var b: Box[Type] = Box(matchCond = {_.isOfType(_)})) extends Type {

  private def maybet: Option[Type] = b.b

  def t: Type = maybet.get

  override val required = true

  override def _toPretty: String = s"?$name"

  override def isRealizable = false

  def _typeCheck(a: Any): Boolean =
    throw new RuntimeException("TVariable is not realizable")

  override def unify(concrete: Type): Boolean = {
    if ((maybet.isDefined || concrete.isRealizable) && cond(concrete))
      b.unify(concrete)
    else
      false
  }

  override def isBound: Boolean = maybet.isEmpty

  override def clear() {
    b.clear()
  }

  override def subst(): Type = {
    assert(maybet.isDefined)
    t
  }

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TVariable is not realizable")

  val ordering: ExtendedOrdering = null
}

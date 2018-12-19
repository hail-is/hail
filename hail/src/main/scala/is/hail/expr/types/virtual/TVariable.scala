package is.hail.expr.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.Box
import is.hail.expr.types.physical.PType

import scala.reflect.ClassTag

final case class TVariable(name: String, cond: (Type) => Boolean = { _ => true }, var b: Box[Type] = Box(matchCond = _.isOfType(_))) extends Type {
  def physicalType: PType = ???

  def t: Type = b.get

  override val required = true

  override def _toPretty: String = s"?$name"

  override def isRealizable = false

  def _typeCheck(a: Any): Boolean =
    throw new RuntimeException("TVariable is not realizable")

  override def unify(concrete: Type): Boolean =
    concrete.isRealizable && cond(concrete) && b.unify(concrete)

  override def isBound: Boolean = b.isEmpty

  override def clear() {
    b.clear()
  }

  override def subst(): Type = {
    assert(b.isDefined)
    t
  }

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TVariable is not realizable")

  val ordering: ExtendedOrdering = null
}

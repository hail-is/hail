package is.hail.expr.typ

import is.hail.annotations._
import is.hail.expr._

import scala.reflect.ClassTag

/**
  * Created by dking on 12/21/17.
  */
final case class TAggregableVariable(elementType: Type, st: Box[SymbolTable]) extends Type {
  override val required = true

  def _toString = s"?Aggregable[$elementType]"

  override def isRealizable = false

  override def children = Seq(elementType)

  def _typeCheck(a: Any): Boolean =
    throw new RuntimeException("TAggregableVariable is not realizable")

  override def unify(concrete: Type): Boolean = concrete match {
    case cagg: TAggregable =>
      elementType.unify(cagg.elementType) && st.unify(cagg.symTab)
    case _ => false
  }

  override def isBound: Boolean = elementType.isBound & st.b.nonEmpty

  override def clear() {
    st.clear()
  }

  override def subst(): Type = {
    assert(st != null)
    TAggregable(elementType.subst(), st.get)
  }

  override def desc: String = TAggregable.desc

  override def canCompare(other: Type): Boolean = false

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TAggregableVariable is not realizable")

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    throw new RuntimeException("TAggregableVariable is not realizable")
}

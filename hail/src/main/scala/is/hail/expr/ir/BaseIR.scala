package is.hail.expr.ir

import is.hail.expr.types.BaseType

abstract class BaseIR {
  def typ: BaseType

  def children: IndexedSeq[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR

  def deepCopy(): this.type = copy(newChildren = children.map(_.deepCopy())).asInstanceOf[this.type]

  lazy val noSharing: this.type = if (HasIRSharing(this)) this.deepCopy() else this

  def mapChildren(f: (BaseIR) => BaseIR): BaseIR = {
    val newChildren = children.map(f)
    if ((children, newChildren).zipped.forall(_ eq _))
      this
    else
      copy(newChildren)
  }
}

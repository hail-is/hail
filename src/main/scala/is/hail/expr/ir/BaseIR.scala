package is.hail.expr.ir

import is.hail.expr.types.BaseType

/**
  * Created by cotton on 6/26/18.
  */
abstract class BaseIR {
  def typ: BaseType

  def children: IndexedSeq[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR

  def deepCopy(): this.type = copy(newChildren = children.map(_.deepCopy())).asInstanceOf[this.type]

  def mapChildren(f: (BaseIR) => BaseIR): BaseIR = {
    copy(children.map(f))
  }
}

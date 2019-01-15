package is.hail.expr.ir

import java.io.{ObjectInputStream, ObjectOutputStream}

import is.hail.expr.types.BaseType

abstract class BaseIR {
  def typ: BaseType

  def children: IndexedSeq[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR

  def deepCopy(): this.type = copy(newChildren = children.map(_.deepCopy())).asInstanceOf[this.type]

  def mapChildren(f: (BaseIR) => BaseIR): BaseIR = {
    val newChildren = children.map(f)
    if ((children, newChildren).zipped.forall(_ eq _))
      this
    else
      copy(newChildren)
  }

  private final def writeObject(out: ObjectOutputStream): Unit = throw new UnsupportedOperationException(this.toString)

  private final def readObject(in: ObjectInputStream): Unit = throw new UnsupportedOperationException(this.toString)
}

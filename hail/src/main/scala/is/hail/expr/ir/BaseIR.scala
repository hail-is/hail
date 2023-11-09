package is.hail.expr.ir

import is.hail.types.BaseType
import is.hail.types.virtual.Type
import is.hail.utils.StackSafe._
import is.hail.utils._

abstract class BaseIR {
  def typ: BaseType

  protected def childrenSeq: IndexedSeq[BaseIR]

  def children: Iterable[BaseIR] = childrenSeq

  protected def copy(newChildren: IndexedSeq[BaseIR]): BaseIR

  def deepCopy(): this.type = copy(newChildren = childrenSeq.map(_.deepCopy())).asInstanceOf[this.type]

  lazy val noSharing: this.type = if (HasIRSharing(this)) this.deepCopy() else this

  def mapChildrenWithIndex(f: (BaseIR, Int) => BaseIR): BaseIR = {
    val newChildren = childrenSeq.view.zipWithIndex.map(f.tupled).toArray
    if (childrenSeq.elementsSameObjects(newChildren))
      this
    else
      copy(newChildren)
  }

  def mapChildren(f: (BaseIR) => BaseIR): BaseIR = {
    val newChildren = childrenSeq.map(f)
    if (childrenSeq.elementsSameObjects(newChildren))
      this
    else
      copy(newChildren)
  }

  def mapChildrenWithIndexStackSafe(f: (BaseIR, Int) => StackFrame[BaseIR]): StackFrame[BaseIR] = {
    call(childrenSeq.iterator.zipWithIndex.map(f.tupled).collectRecur).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copy(newChildren)
    }
  }

  def mapChildrenStackSafe(f: BaseIR => StackFrame[BaseIR]): StackFrame[BaseIR] = {
    call(childrenSeq.mapRecur(f)).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copy(newChildren)
    }
  }

  def forEachChildWithEnv(env: BindingEnv[Type])(f: (BaseIR, BindingEnv[Type]) => Unit): Unit = {
    childrenSeq.view.zipWithIndex.foreach { case (child, i) =>
      val childEnv = ChildBindings(this, i, env)
      f(child, childEnv)
    }
  }

  def mapChildrenWithEnv(env: BindingEnv[Type])(f: (BaseIR, BindingEnv[Type]) => BaseIR): BaseIR = {
    val newChildren = childrenSeq.toArray
    var res = this
    for (i <- newChildren.indices) {
      val childEnv = ChildBindings(res, i, env)
      val child = newChildren(i)
      val newChild = f(child, childEnv)
      if (!(newChild eq child)) {
        newChildren(i) = newChild
        res = res.copy(newChildren)
      }
    }
    res
  }

  def forEachChildWithEnvStackSafe(env: BindingEnv[Type])(f: (BaseIR, Int, BindingEnv[Type]) => StackFrame[Unit]): StackFrame[Unit] = {
    childrenSeq.view.zipWithIndex.foreachRecur { case (child, i) =>
      val childEnv = ChildBindings(this, i, env)
      f(child, i, childEnv)
    }
  }
}

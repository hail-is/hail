package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.types.BaseType
import is.hail.types.virtual.Type
import is.hail.utils._
import is.hail.utils.StackSafe._

case class Name(str: String) {
  override def toString: String = str
}

abstract class BaseIR {
  def typ: BaseType

  protected def childrenSeq: IndexedSeq[BaseIR]

  def children: Iterable[BaseIR] = childrenSeq

  protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BaseIR

  def deepCopy(): this.type =
    copyWithNewChildren(newChildren = childrenSeq.map(_.deepCopy())).asInstanceOf[this.type]

  def noSharing(ctx: ExecuteContext): this.type =
    if (HasIRSharing(ctx)(this)) this.deepCopy() else this

  // For use as a boolean flag by IR passes. Each pass uses a different sentinel value to encode
  // "true" (and anything else is false). As long as we maintain the global invariant that no
  // two passes use the same sentinel value, this allows us to reuse this field across passes
  // without ever having to initialize it at the start of a pass.
  // New sentinel values can be obtained by `nextFlag` on `IRMetadata`.
  var mark: Int = 0

  def isAlphaEquiv(ctx: ExecuteContext, other: BaseIR): Boolean =
    /* FIXME: rewrite to not rebuild the irs, by maintaining an env mapping left names to right
     * names */
    NormalizeNames(ctx, this, allowFreeVariables = true) ==
      NormalizeNames(ctx, other, allowFreeVariables = true)

  def mapChildrenWithIndex(f: (BaseIR, Int) => BaseIR): BaseIR = {
    val newChildren = childrenSeq.view.zipWithIndex.map(f.tupled).toArray
    if (childrenSeq.elementsSameObjects(newChildren))
      this
    else
      copyWithNewChildren(newChildren)
  }

  def mapChildren(f: (BaseIR) => BaseIR): BaseIR = {
    val newChildren = childrenSeq.map(f)
    if (childrenSeq.elementsSameObjects(newChildren))
      this
    else
      copyWithNewChildren(newChildren)
  }

  def mapChildrenWithIndexStackSafe(f: (BaseIR, Int) => StackFrame[BaseIR]): StackFrame[BaseIR] = {
    call(childrenSeq.iterator.zipWithIndex.map(f.tupled).collectRecur).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copyWithNewChildren(newChildren)
    }
  }

  def mapChildrenStackSafe(f: BaseIR => StackFrame[BaseIR]): StackFrame[BaseIR] = {
    call(childrenSeq.mapRecur(f)).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copyWithNewChildren(newChildren)
    }
  }

  def forEachChildWithEnv[E <: GenericBindingEnv[E, Type]](env: E)(f: (BaseIR, E) => Unit): Unit =
    childrenSeq.view.zipWithIndex.foreach { case (child, i) =>
      val childEnv = env.extend(Bindings.get(this, i))
      f(child, childEnv)
    }

  def mapChildrenWithEnv(env: BindingEnv[Type])(f: (BaseIR, BindingEnv[Type]) => BaseIR): BaseIR =
    mapChildrenWithEnv[BindingEnv[Type]](env, (env, bindings) => env.extend(bindings))(f)

  def mapChildrenWithEnv[E](
    env: E,
    update: (E, Bindings[Type]) => E,
  )(
    f: (BaseIR, E) => BaseIR
  ): BaseIR = {
    val newChildren = Array(childrenSeq: _*)
    var res = this
    for (i <- newChildren.indices) {
      val childEnv = update(env, Bindings.get(res, i))
      val child = newChildren(i)
      val newChild = f(child, childEnv)
      if (!(newChild eq child)) {
        newChildren(i) = newChild
        res = res.copyWithNewChildren(newChildren)
      }
    }
    res
  }

  def mapChildrenWithEnvStackSafe[E](
    env: E,
    update: (E, Bindings[Type]) => E,
  )(
    f: (BaseIR, E) => StackFrame[BaseIR]
  ): StackFrame[BaseIR] = {
    val newChildren = Array(childrenSeq: _*)
    var res = this
    newChildren.indices.foreachRecur { i =>
      val childEnv = update(env, Bindings.get(res, i))
      val child = newChildren(i)
      f(child, childEnv).map { newChild =>
        if (!(newChild eq child)) {
          newChildren(i) = newChild
          res = res.copyWithNewChildren(newChildren)
        }
      }
    }.map(_ => res)
  }

  def forEachChildWithEnvStackSafe[E <: GenericBindingEnv[E, Type]](
    env: E
  )(
    f: (BaseIR, Int, E) => StackFrame[Unit]
  ): StackFrame[Unit] =
    childrenSeq.view.zipWithIndex.foreachRecur { case (child, i) =>
      val childEnv = env.extend(Bindings.get(this, i))
      f(child, i, childEnv)
    }
}

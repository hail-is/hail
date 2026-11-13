package is.hail.expr.ir

import is.hail.collection.implicits.toRichIndexedSeqAnyRef
import is.hail.expr.ir.defs.Block
import is.hail.types.virtual.{Type, VType}
import is.hail.utils.StackSafe._

import scala.collection.immutable.ArraySeq

case class Name(str: String) {
  override def toString: String = str
}

/** @param tag
  *   identifies this node's class, for dispatch with `@switch`. See [[Tag]]. It is a field, not a
  *   method, so reading it is a load rather than a virtual call.
  */
abstract class BaseIR(final val tag: Tag) {
  def typ: VType

  protected def childrenSeq: IndexedSeq[BaseIR]

  def children: Iterable[BaseIR] = childrenSeq

  def getChild(idx: Int): BaseIR = childrenSeq(idx)

  protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BaseIR

  def deepCopy: BaseIR =
    copyWithNewChildren(newChildren = childrenSeq.map(_.deepCopy))

  // For use as a boolean flag by IR passes. Each pass uses a different sentinel value to encode
  // "true" (and anything else is false). As long as we maintain the global invariant that no
  // two passes use the same sentinel value, this allows us to reuse this field across passes
  // without ever having to initialize it at the start of a pass.
  // New sentinel values can be obtained by `nextFlag` on `IRMetadata`.
  var mark: Int = 0

  def mapChildrenWithIndex(f: (BaseIR, Int) => BaseIR): BaseIR = {
    val newChildren = childrenSeq.view.zipWithIndex.map(f.tupled).to(ArraySeq)
    if (childrenSeq.elementsSameObjects(newChildren))
      this
    else
      copyWithNewChildren(newChildren)
  }

  def mapChildren(f: BaseIR => BaseIR): BaseIR = {
    val newChildren = childrenSeq.map(f)
    if (childrenSeq.elementsSameObjects(newChildren))
      this
    else
      copyWithNewChildren(newChildren)
  }

  def mapChildrenWithIndexStackSafe(f: (BaseIR, Int) => StackFrame[BaseIR]): StackFrame[BaseIR] = {
    childrenSeq.iterator.zipWithIndex.map(f.tupled).collectRecur.map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copyWithNewChildren(newChildren)
    }
  }

  def mapChildrenStackSafe(f: BaseIR => StackFrame[BaseIR]): StackFrame[BaseIR] = {
    childrenSeq.mapRecur(f).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copyWithNewChildren(newChildren)
    }
  }

  def forEachChildWithEnv[E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, E[A]) => Unit
  ): Unit =
    this match {
      case b: Block =>
        ChildEnv.walkBlock(b, env)(v) { (c, _, ce) => f(c, ce); c }
      case _ =>
        for (i <- childrenSeq.indices)
          f(childrenSeq(i), ChildEnv.get(env, this, childrenSeq, i)(v))
    }

  def mapChildrenWithEnv[E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, E[A]) => BaseIR
  ): BaseIR = {
    val newChildren = childrenSeq.toArray
    val cs = ArraySeq.unsafeWrapArray(newChildren)
    this match {
      case b: Block =>
        ChildEnv.walkBlock(b, env)(v) { (c, i, ce) => newChildren(i) = f(c, ce); newChildren(i) }
      case _ =>
        for (i <- newChildren.indices)
          newChildren(i) = f(newChildren(i), ChildEnv.get(env, this, cs, i)(v))
    }
    if (childrenSeq.elementsSameObjects(cs)) this else copyWithNewChildren(cs)
  }

  def mapChildrenWithEnvStackSafe[E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, E[A]) => StackFrame[BaseIR]
  ): StackFrame[BaseIR] = {
    val newChildren = childrenSeq.toArray
    val cs = ArraySeq.unsafeWrapArray(newChildren)

    val walk =
      this match {
        case b: Block =>
          ChildEnv.walkBlockStackSafe(b, env)(v) { (c, i, ce) =>
            f(c, ce).map { c2 => newChildren(i) = c2; c2 }
          }
        case _ =>
          newChildren.indices.foreachRecur { i =>
            f(newChildren(i), ChildEnv.get(env, this, cs, i)(v)).map(newChildren(i) = _)
          }
      }

    walk.map(_ => if (childrenSeq.elementsSameObjects(cs)) this else copyWithNewChildren(cs))
  }

  def forEachChildWithEnvStackSafe[E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, Int, E[A]) => StackFrame[Unit]
  ): StackFrame[Unit] =
    this match {
      case b: Block =>
        ChildEnv.walkBlockStackSafe(b, env)(v)((c, i, ce) => f(c, i, ce).map(_ => c))
      case _ =>
        childrenSeq.indices.foreachRecur { i =>
          f(childrenSeq(i), i, ChildEnv.get(env, this, childrenSeq, i)(v))
        }
    }
}

trait PreservesRowsOrCols extends BaseIR {
  def preservesRowsOrColsFrom: BaseIR
}

object PreservesOrRemovesRows {
  def unapply(ir: BaseIR): Option[BaseIR] = ir match {
    case ir: PreservesRows if ir.preservesRowsCond => Some(ir.preservesRowsOrColsFrom)
    case _ => None
  }
}

/** A Table or MatrixTable node should implement this when each row in the `preservesRowsOrColsFrom`
  * input is either deleted or corresponds to a row in the output with the same key. In particular,
  * the total number of rows cannot increase.
  *
  * If `preservesPartitioning` is true, then moreover each surviving input row stays in the same
  * partition.
  */
trait PreservesOrRemovesRows extends PreservesRowsOrCols {
  // Allows the trait to be conditionally applied
  def preservesRowsCond: Boolean = true
  def preservesPartitioning: Boolean = true
}

object PreservesRows {
  def unapply(ir: BaseIR): Option[(BaseIR, Boolean)] = ir match {
    case ir: PreservesRows if ir.preservesRowsCond =>
      Some((ir.preservesRowsOrColsFrom, ir.preservesPartitioning))
    case _ => None
  }
}

/** A Table or MatrixTable implementing this has all the properties of `PreservesOrRemovesRows`, but
  * input rows are never deleted. In particular, the total number of rows is preserved exactly.
  *
  * If `preservesPartitioning` is true, then partition counts are also preserved exactly.
  */
trait PreservesRows extends PreservesOrRemovesRows

object PreservesCols {
  def unapply(ir: BaseIR): Option[BaseIR] = ir match {
    case ir: PreservesCols => Some(ir.preservesRowsOrColsFrom)
    case _ => None
  }
}

/** A MatrixTable should implement this when each column in `preservesRowsOrColsFrom` corresponds to
  * a column in the output with the same key. In particular, the number of columns is preserved
  * exactly.
  */
trait PreservesCols extends PreservesRowsOrCols

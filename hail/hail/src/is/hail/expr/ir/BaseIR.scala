package is.hail.expr.ir

import is.hail.collection.implicits.{rangeTraverseOps, toRichIndexedSeqAnyRef}
import is.hail.types.virtual.{Type, VType}

import scala.collection.immutable.ArraySeq

import cats.{Id, Monad}
import cats.implicits.toFunctorOps

case class Name(str: String) {
  override def toString: String = str
}

abstract class BaseIR {
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

  def mapChildrenM[F[_]: Monad](f: BaseIR => F[BaseIR]): F[BaseIR] = {
    val newChildren = childrenSeq.toArray
    val cs = ArraySeq.unsafeWrapArray(newChildren)
    newChildren.indices.foldM(())((_, i) => f(newChildren(i)).map(c => newChildren(i) = c)).map(
      _ => if (childrenSeq.elementsSameObjects(cs)) this else copyWithNewChildren(cs)
    )
  }

  def forEachChildWithEnv[E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, E[A]) => Unit
  ): Unit = {
    var e = env
    for (i <- childrenSeq.indices)
      e = ChildEnv.step[Id, E, A](e, this, childrenSeq, i)(v) { ce =>
        val child = childrenSeq(i)
        f(child, ce)
        child
      }
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
    var e = env
    for (i <- newChildren.indices)
      e = ChildEnv.step[Id, E, A](e, this, cs, i)(v) { ce =>
        newChildren(i) = f(newChildren(i), ce)
        newChildren(i)
      }
    if (childrenSeq.elementsSameObjects(cs)) this else copyWithNewChildren(cs)
  }

  def mapChildrenWithEnvM[F[_]: Monad, E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, E[A]) => F[BaseIR]
  ): F[BaseIR] = {
    val newChildren = childrenSeq.toArray
    val cs = ArraySeq.unsafeWrapArray(newChildren)
    newChildren.indices.foldM(env) { (e, i) =>
      ChildEnv.step[F, E, A](e, this, cs, i)(v) { ce =>
        f(newChildren(i), ce).map { c => newChildren(i) = c; c }
      }
    }.map(_ => if (childrenSeq.elementsSameObjects(cs)) this else copyWithNewChildren(cs))
  }

  def forEachChildWithEnvM[F[_]: Monad, E[_]: Scopes, A](
    env: E[A]
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, Int, E[A]) => F[Unit]
  ): F[Unit] =
    childrenSeq.indices.foldM(env) { (e, i) =>
      val child = childrenSeq(i)
      ChildEnv.step[F, E, A](e, this, childrenSeq, i)(v)(ce => f(child, i, ce).map(_ => child))
    }.as(())
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

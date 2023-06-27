package is.hail.expr.ir

import cats.Applicative
import cats.implicits.toFunctorOps
import cats.syntax.traverse._
import is.hail.types.BaseType
import is.hail.utils._

import scala.language.higherKinds

abstract class BaseIR {
  def typ: BaseType

  protected def childrenSeq: IndexedSeq[BaseIR]

  def children: Iterable[BaseIR] = childrenSeq
  def numChildren: Int = childrenSeq.length

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR

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

  def traverseChildrenWithIndex[F[_]: Applicative](f: (BaseIR, Int) => F[BaseIR]): F[BaseIR] =
    childrenSeq.zipWithIndex.traverse(f.tupled).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copy(newChildren)
    }

  def traverseChildren[F[_]: Applicative](f: BaseIR => F[BaseIR]): F[BaseIR] =
    childrenSeq.traverse(f).map { newChildren =>
      if (childrenSeq.elementsSameObjects(newChildren))
        this
      else
        copy(newChildren)
    }
}

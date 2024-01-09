package is.hail.expr.ir

import is.hail.backend.ExecuteContext

import scala.collection.mutable

object RefEquality {
  def apply[T <: AnyRef](t: T): RefEquality[T] = new RefEquality[T](t)
}

class RefEquality[+T <: AnyRef](val t: T) {
  override def equals(obj: scala.Any): Boolean = obj match {
    case r: RefEquality[T] => t.eq(r.t)
    case _ => false
  }

  override def hashCode(): Int = System.identityHashCode(t)

  override def toString: String = s"RefEquality($t)"
}

object Memo {
  def empty[T]: Memo[T] = new Memo[T](new mutable.HashMap[RefEquality[BaseIR], T])
}

class Memo[T] private (val m: mutable.HashMap[RefEquality[BaseIR], T]) {
  def bind(ir: BaseIR, t: T): Memo[T] = bind(RefEquality(ir), t)

  def bind(ir: RefEquality[BaseIR], t: T): Memo[T] = {
    if (m.contains(ir))
      throw new RuntimeException(s"IR already in memo: ${ir.t}")
    m += ir -> t
    this
  }

  def bindIf(test: Boolean, ir: BaseIR, t: => T): Memo[T] =
    if (test) bind(ir, t) else this

  def contains(ir: BaseIR): Boolean = contains(RefEquality(ir))
  def contains(ir: RefEquality[BaseIR]): Boolean = m.contains(ir)

  def lookup(ir: BaseIR): T = lookup(RefEquality(ir))
  def lookup(ir: RefEquality[BaseIR]): T = m(ir)

  def apply(ir: BaseIR): T = lookup(ir)

  def update(ir: BaseIR, t: => T): Unit = m.update(RefEquality(ir), t)

  def get(ir: BaseIR): Option[T] = get(RefEquality(ir))
  def get(ir: RefEquality[BaseIR]): Option[T] = m.get(ir)

  def getOrElse(ir: BaseIR, default: => T): T = m.getOrElse(RefEquality(ir), default)

  def getOrElseUpdate(ir: BaseIR, t: => T): T = m.getOrElseUpdate(RefEquality(ir), t)

  def getOrElseUpdate(ir: RefEquality[BaseIR], t: => T): T = m.getOrElseUpdate(ir, t)

  def delete(ir: BaseIR): Unit = delete(RefEquality(ir))
  def delete(ir: RefEquality[BaseIR]): Unit = m -= ir

  override def toString: String = s"Memo($m)"
}

object HasIRSharing {
  def apply(ctx: ExecuteContext)(ir: BaseIR): Boolean = {
    val mark = ctx.irMetadata.nextFlag

    for (node <- IRTraversal.levelOrder(ir)) {
      if (node.mark == mark) return true
      node.mark = mark
    }

    false
  }
}

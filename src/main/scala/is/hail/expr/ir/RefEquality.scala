package is.hail.expr.ir

import scala.collection.mutable

object RefEquality {
  def apply[T <: AnyRef](t: T): RefEquality[T] = new RefEquality[T](t)
}

class RefEquality[T <: AnyRef](val t: T) {
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

class Memo[T] private(val m: mutable.HashMap[RefEquality[BaseIR], T]) {
  def bind(ir: BaseIR, t: T): Memo[T] = {
    val re = RefEquality(ir)
    assert(!m.contains(re))
    m += re -> t
    this
  }

  def lookup(ir: BaseIR): T = m(RefEquality(ir))

  def get(ir: BaseIR): Option[T] = m.get(RefEquality(ir))

}


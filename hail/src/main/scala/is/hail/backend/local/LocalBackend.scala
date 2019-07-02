package is.hail.backend.local

import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.ir._
import is.hail.utils.{ExecutionTimer, Timings}

import scala.reflect.ClassTag

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T]

object LocalBackend extends Backend {

  def broadcast[T: ClassTag](value: T): LocalBroadcastValue[T] = new LocalBroadcastValue(value)

  def parallelizeAndComputeWithIndex[T : ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    collection.zipWithIndex.map { case (elt, i) => f(elt, i) }.toArray
  }

  override def lower(ir: IR, timer: Option[ExecutionTimer], optimize: Boolean): IR = LowerTableIR(ir, timer, optimize)
}

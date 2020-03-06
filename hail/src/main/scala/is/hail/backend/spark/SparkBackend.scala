package is.hail.backend.spark

import is.hail.HailContext
import is.hail.backend.{Backend, BroadcastValue, HailTaskContext}
import is.hail.expr.ir._
import is.hail.utils._
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

object SparkBackend {
  def executeJSON(ir: IR): String = HailContext.backend.executeJSON(ir)
}

class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

class SparkTaskContext(ctx: TaskContext) extends HailTaskContext {
  type BackendType = SparkBackend
  override def stageId(): Int = ctx.stageId()
  override def partitionId(): Int = ctx.partitionId()
  override def attemptNumber(): Int = ctx.attemptNumber()
}

case class SparkBackend(sc: SparkContext) extends Backend {

  override val cache: SparkValueCache = SparkValueCache()

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new SparkBroadcastValue[T](sc.broadcast(value))

  def parallelizeAndComputeWithIndex[T : ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val rdd = sc.parallelize[T](collection, numSlices = collection.length)
    rdd.mapPartitionsWithIndex { (i, it) =>
      HailTaskContext.setTaskContext(new SparkTaskContext(TaskContext.get))
      val elt = it.next()
      assert(!it.hasNext)
      Iterator.single(f(elt, i))
    }.collect()
  }

  override def asSpark(): SparkBackend = this
}

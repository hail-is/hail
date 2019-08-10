package is.hail.backend.distributed

import is.hail.backend.{Backend, BroadcastValue}
import is.hail.scheduler._
import org.apache.hadoop.conf.Configuration

import scala.reflect.ClassTag

class DistributedBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

class DistributedBackend(hostname: String, hconf: Configuration) extends Backend {

  lazy val scheduler: SchedulerAppClient = new SchedulerAppClient(hostname)

  def broadcast[T: ClassTag](value: T): DistributedBroadcastValue[T] = new DistributedBroadcastValue[T](value)

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val da = new DArray[U] with Serializable {
      type Context = (T, Int)

      val contexts: Array[Context] = collection.zipWithIndex
      val body: Context => U = { case (t, i) => f(t, i) }
    }

    val result = new Array[U](collection.length)
    scheduler.submit(da, { (i: Int, u: U) => result.update(i, u) })
    result
  }
}

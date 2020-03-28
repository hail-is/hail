package is.hail.backend

import is.hail.backend.spark.SparkBackend
import is.hail.utils._

import scala.reflect.ClassTag

abstract class BroadcastValue[T] { def value: T }

abstract class Backend {
  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U]

  def stop(): Unit

  def asSpark(): SparkBackend = fatal("SparkBackend needed for this operation.")
}

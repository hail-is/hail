package is.hail.backend.service

import is.hail.backend.{Backend, BroadcastValue}

import scala.reflect.ClassTag

object ServiceBackend extends Backend {
  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val n = collection.length
    val r = new Array[U](n)
    var i = 0
    while (i < n) {
      r(i) = f(collection(i), i)
      i += 1
    }
    r
  }

  def stop(): Unit = ()

  def apply(): ServiceBackend = {
    new ServiceBackend()
  }
}

class ServiceBackend() {
  def request(): Int = 5
}

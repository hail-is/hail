package is.hail.methods

import is.hail.expr._
import is.hail.expr.types.virtual.Type

import scala.collection.mutable.ArrayBuffer

class QuantilesAggregator[T, +S](bufSize: Int, t: Type) extends TypedAggregator[S] {

  val ord = t.ordering
  var bufferPool: List[Array[T]] = Nil

  val fullBuffers: ArrayBuffer[Array[T]] = ArrayBuffer(null)
  var inputBuffer: Array[T] = Array.ofDim[T](bufSize)
  var inBufSize: Int = 0

  def seqOp(x: Any): Unit = {
    inputBuffer(inBufSize) = x.asInstanceOf[T]
    inBufSize += 1
    if (inBufSize == bufSize) {
      if (fullBuffers(0) == null) {
        fullBuffers(0) = inputBuffer
        inputBuffer = getBuffer()
        inBufSize = 0
      } else {

      }
    }
  }

  def compact(height: Int, right: Array[T], skipFirst: Boolean): Unit = {
    if (fullBuffers.size == height) {
      fullBuffers += null
    }
    val left = fullBuffers(height)
    fullBuffers(height) = null
    var i: Int = 0
    var j: Int = 0
    val out = getBuffer()
    var o: Int = 0

    if (skipFirst) {
      if (ord.lt(left(i), right(j))) i += 1 else j += 1
    }
    while (i != bufSize && j != bufSize) {
      if (ord.lt(left(i), right(j))) {
        out(o) = left(i)
        i += 1
      } else {
        out(o) = right(j)
        j += 1
      }
      if (ord.lt(left(i), right(j))) {
        i += 1
      } else {
        j += 1
      }
    }
    while (i != bufSize) {

    }
  }

  def getBuffer(): Array[T] = bufferPool match {
    case Nil =>
      Array.ofDim[T](bufSize)
    case buf :: buffs =>
      bufferPool = buffs
      buf
  }

  def combOp(agg2: this.type): Unit

  def result: S

  def copy(): QuantilesAggregator[T, S] = new QuantilesAggregator[T, S](bufSize)
}

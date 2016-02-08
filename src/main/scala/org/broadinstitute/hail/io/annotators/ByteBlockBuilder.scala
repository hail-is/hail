package org.broadinstitute.hail.io.annotators

import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.variant.LZ4Utils

import scala.collection.mutable
import scala.reflect.ClassTag

class ByteBlockBuilder[T](ser: SerializerInstance, blockSize: Int = 10)(implicit ev: ClassTag[T]) {
  val compressedArrays = mutable.ArrayBuilder.make[(Int, Array[Byte])]
  var buildingBlock = mutable.ArrayBuilder.make[T]
  var count = 0
  var index = 0


  def add(element: T): (Int, Int) = {
    buildingBlock += element
    val innerIndex = count
    count += 1

    if (count >= blockSize) {
      val res = ser.serialize(buildingBlock.result()).array()
      compressedArrays += ((res.length, LZ4Utils.compress(res)))
      buildingBlock.clear()
      count = 0
      index += 1
      (index - 1, innerIndex)
    }
    else
      (index, innerIndex)
  }

  def result(): Array[(Int, Array[Byte])] = {
    val res = ser.serialize(buildingBlock.result()).array()
    compressedArrays += ((res.length, LZ4Utils.compress(res)))
    val compressedResult = compressedArrays.result()
    compressedArrays.clear()
    compressedResult
  }
}

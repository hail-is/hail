package org.broadinstitute.hail.io

class ByteBlock extends Serializable {
  private var byteArray: Option[Array[Byte]] = None
  private var index: Option[Int] = None

  def setBlock(arr: Array[Byte]) = byteArray = Some(arr)

  def setIndex(ind: Int) = index = Some(ind)


  def getArray: Array[Byte] = {
    byteArray match {
      case Some(arr) => arr
      case None => throw new IllegalAccessError("getArray was called from a byteBlock whose array was not set")
    }
  }

  def getIndex: Int = {
    index match {
      case Some(ind) => ind
      case None => throw new NullPointerException("getIndex was called from a byteBlock whose index was not set")
    }
  }

  def length: Long = {
    byteArray match {
      case Some(arr) => arr.length
      case None => throw new IllegalAccessError("length was called from a byteBlock whose array was not set")
    }
  }
}

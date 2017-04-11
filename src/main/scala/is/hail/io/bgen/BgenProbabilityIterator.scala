package is.hail.io.bgen

class BgenProbabilityIterator(input: Array[Byte], nBitsPerProb: Int, start: Int = 0) extends Iterator[Int] {
  val nBytes = input.length
  val bitMask = ~0L >>> (64 - nBitsPerProb)

  var byteIndex = start
  var data = 0L
  var dataSize = 0

  override def next(): Int = {
    while (dataSize < nBitsPerProb && byteIndex < nBytes) {
      data |= ((input(byteIndex) & 0xffL) << dataSize)
      byteIndex += 1
      dataSize += 8
    }

    val result = data & bitMask
    dataSize -= nBitsPerProb
    data = data >>> nBitsPerProb
    result.toInt
  }

  override def hasNext: Boolean = (byteIndex < nBytes) || (dataSize >= nBitsPerProb)
}

package is.hail.io.bgen

import is.hail.io.ByteArrayReader
import is.hail.utils.UInt
import is.hail.utils._

class BgenProbabilityIterator(input: ByteArrayReader, nBitsPerProb: Int) extends HailIterator[UInt] {
  val bitMask = (1L << nBitsPerProb) - 1
  var data = 0L
  var dataSize = 0

  override def next(): UInt = {
    while (dataSize < nBitsPerProb && input.hasNext()) {
      data |= ((input.read() & 0xffL) << dataSize)
      dataSize += 8
    }
    assert(dataSize >= nBitsPerProb, s"Data size `$dataSize' less than nBitsPerProb `$nBitsPerProb'.")

    val result = data & bitMask
    dataSize -= nBitsPerProb
    data = data >>> nBitsPerProb

    result.toUInt
  }

  override def hasNext: Boolean = input.hasNext() || (dataSize >= nBitsPerProb)
}

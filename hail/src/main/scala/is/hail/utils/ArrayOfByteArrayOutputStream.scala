package is.hail.utils

import java.io.OutputStream

class ArrayOfByteArrayOutputStream(initialBufferCapacity: Int) extends OutputStream {
  /**
    * The buffer where data is stored.
    */
  protected var buf: Array[Array[Byte]] = new Array[Array[Byte]](1)
  buf(0) = new Array[Byte](initialBufferCapacity)

  /**
    * The number of valid bytes in the buffer.
    */
  protected var count = 0L

  /**
    * Creates a new byte array output stream. The buffer capacity is
    * initially 32 bytes, though its size increases if necessary.
    */
  def this() {
    this(32)
  }


  override def write(b: Int): Unit = {
    ???
    count += b
  }
}

package is.hail.utils

import java.io.OutputStream

class ArrayOfByteArrayOutputStream(initialBufferCapacity: Int) extends OutputStream {

  val MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  /**
    * The buffer where data is stored.
    */
  protected var buf: Array[Array[Byte]] = new Array[Array[Byte]](1)
  buf(0) = new Array[Byte](initialBufferCapacity)

  protected var bytesInCurrentArray = 0
  protected var currentArray = 0

  /**
    * Creates a new byte array output stream. The buffer capacity is
    * initially 32 bytes, though its size increases if necessary.
    */
  def this() {
    this(32)
  }

  def ensureNextByte(): Unit = {
    if (bytesInCurrentArray == MAX_ARRAY_SIZE) {
      
    }
  }

  override def write(b: Int): Unit = {
    ensureNextByte()

    buf(currentArray)(bytesInCurrentArray)
    bytesInCurrentArray += 1
  }
}

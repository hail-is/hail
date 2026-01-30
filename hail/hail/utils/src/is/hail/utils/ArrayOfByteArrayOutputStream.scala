package is.hail.utils

import java.io.{ByteArrayOutputStream, OutputStream}

class ArrayOfByteArrayOutputStream(initialBufferCapacity: Int) extends OutputStream {

  val MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  /** The buffer where data is stored. */
  protected var buf = Array.newBuilder[ByteArrayOutputStream]
  protected var currentArray = new ByteArrayOutputStream(initialBufferCapacity)
  buf += currentArray

  protected var bytesInCurrentArray = 0

  /** Creates a new byte array output stream. The buffer capacity is initially 32 bytes, though its
    * size increases if necessary.
    */
  def this() =
    this(32)

  def ensureNextByte(): Unit =
    if (bytesInCurrentArray == MAX_ARRAY_SIZE) {
      currentArray = new ByteArrayOutputStream(initialBufferCapacity)
      buf += currentArray
      bytesInCurrentArray = 0
    }

  override def write(b: Int): Unit = {
    ensureNextByte()

    currentArray.write(b)
    bytesInCurrentArray += 1
  }

  override def write(b: Array[Byte], off: Int, len: Int): Unit = {
    var bytesWritten = 0

    while (bytesWritten < len) {
      ensureNextByte()
      val remainingBytesAllowed = MAX_ARRAY_SIZE - bytesInCurrentArray
      val bytesToWrite = math.min(remainingBytesAllowed, len - bytesWritten)
      currentArray.write(b, off + bytesWritten, bytesToWrite)
      bytesInCurrentArray += bytesToWrite
      bytesWritten += bytesToWrite
    }
  }

  def toByteArrays(): Array[Array[Byte]] =
    buf.result().map(_.toByteArray)
}

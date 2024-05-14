package is.hail.utils

import java.io.{ByteArrayOutputStream, OutputStream}

class ArrayOfByteArrayOutputStream(initialBufferCapacity: Int) extends OutputStream {

  val MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  /** The buffer where data is stored. */
  protected var buf = new BoxedArrayBuilder[ByteArrayOutputStream](1)
  buf += new ByteArrayOutputStream(initialBufferCapacity)

  protected var bytesInCurrentArray = 0
  protected var currentArray = buf(0)

  /** Creates a new byte array output stream. The buffer capacity is initially 32 bytes, though its
    * size increases if necessary.
    */
  def this() =
    this(32)

  def ensureNextByte(): Unit = {
    if (bytesInCurrentArray == MAX_ARRAY_SIZE) {
      buf.ensureCapacity(buf.length + 1)
      buf += new ByteArrayOutputStream(initialBufferCapacity)
      currentArray = buf.last
      bytesInCurrentArray = 0
    }
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

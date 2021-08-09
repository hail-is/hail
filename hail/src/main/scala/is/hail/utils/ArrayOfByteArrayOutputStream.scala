package is.hail.utils

import java.io.{ByteArrayOutputStream, OutputStream}

class ArrayOfByteArrayOutputStream(initialBufferCapacity: Int) extends OutputStream {

  val MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  /**
    * The buffer where data is stored.
    */
  protected var buf = new BoxedArrayBuilder[ByteArrayOutputStream](1)
  buf += new ByteArrayOutputStream(initialBufferCapacity)

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
      buf.ensureCapacity(buf.length + 1)
      buf += new ByteArrayOutputStream(initialBufferCapacity)
      currentArray += 1
      bytesInCurrentArray = 0
    }
  }

  override def write(b: Int): Unit = {
    ensureNextByte()

    buf(buf.length - 1).write(b)
    bytesInCurrentArray += 1
  }

  def toByteArrays(): Array[Array[Byte]] = {
    buf.result().map(_.toByteArray)
  }
}

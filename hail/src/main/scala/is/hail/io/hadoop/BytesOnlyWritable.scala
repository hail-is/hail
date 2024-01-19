package is.hail.io.hadoop

import java.io.{DataInput, DataOutput}

import org.apache.hadoop.io.Writable

class BytesOnlyWritable(var bytes: Array[Byte]) extends Writable {

  def this() = this(null)

  def set(bytes: Array[Byte]): Unit =
    this.bytes = bytes

  override def write(out: DataOutput): Unit = {
    assert(bytes != null)
    out.write(bytes, 0, bytes.length)
  }

  override def readFields(in: DataInput): Unit =
    throw new UnsupportedOperationException()
}

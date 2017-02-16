package is.hail.io.hadoop

import java.io.DataOutput

import org.apache.hadoop.io.BytesWritable

class BytesOnlyWritable(var bytes: Array[Byte])
  extends BytesWritable(bytes, bytes.length) {

  def this() = this(Array.empty[Byte])

  override def write(out: DataOutput) {
    out.write(super.getBytes, 0, super.getLength)
  }
}

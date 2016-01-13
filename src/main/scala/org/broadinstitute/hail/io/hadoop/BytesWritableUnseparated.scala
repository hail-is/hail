package org.broadinstitute.hail.io.hadoop

import java.io.IOException
import java.io.DataOutput
import org.apache.hadoop.classification.InterfaceAudience
import org.apache.hadoop.classification.InterfaceStability
import org.apache.hadoop.io.BytesWritable

@InterfaceAudience.Public
@InterfaceStability.Stable class BytesWritableUnseparated(var bytes: Array[Byte], var size: Int)
  extends BytesWritable(bytes, size) {
  assert(bytes.length == size)
  val LENGTH_BYTES: Int = 4

  def this() = this(Array.empty[Byte], 0)

  def this(bytes: Array[Byte]) = this(bytes, bytes.length)

  @throws(classOf[IOException])
  override def write(out: DataOutput) {
    out.write(super.getBytes, 0, super.getLength)
  }
}

package is.hail.io

import java.io.{DataInputStream, DataOutputStream, InputStream, OutputStream}

import org.apache.hadoop.fs.PathIOException

package object fs {
  type PositionedInputStream = InputStream with Positioned

  type SeekableInputStream = InputStream with Seekable

  type SeekableDataInputStream = DataInputStream with Seekable

  type PositionedOutputStream = OutputStream with Positioned

  type PositionedDataOutputStream = DataOutputStream with Positioned

  def outputStreamToPositionedDataOutputStream(os: OutputStream): PositionedDataOutputStream =
    new WrappedPositionedDataOutputStream(
      new WrappedPositionOutputStream(
        os
      )
    )

  private[this] val PartRegex = """.*/?part-(\d+).*""".r

  def getPartNumber(fname: String): Int =
    fname match {
      case PartRegex(i) => i.toInt
      case _ => throw new PathIOException(s"invalid partition file '$fname'")
    }

  private[this] val Kilo: Long = 1024
  private[this] val Mega: Long = Kilo * 1024
  private[this] val Giga: Long = Mega * 1024
  private[this] val Tera: Long = Giga * 1024

  def readableBytes(bytes: Long): String =
    if (bytes < Kilo) bytes.toString
    else if (bytes < Mega) formatDigits(bytes, Kilo) + "K"
    else if (bytes < Giga) formatDigits(bytes, Mega) + "M"
    else if (bytes < Tera) formatDigits(bytes, Giga) + "G"
    else formatDigits(bytes, Tera) + "T"

  private def formatDigits(n: Long, factor: Long): String =
    "%.1f".format(n / factor.toDouble)
}

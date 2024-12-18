package is.hail.io

import java.io.{DataInputStream, DataOutputStream, InputStream, OutputStream}

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
}

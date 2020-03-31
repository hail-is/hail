package is.hail.io

import java.io.{DataInputStream, DataOutputStream, InputStream, OutputStream}

package object fs {
  type SeekableInputStream = InputStream with Seekable

  type SeekableDataInputStream = DataInputStream with Seekable

  type PositionedOutputStream = OutputStream with Positioned

  type PositionedDataOutputStream = DataOutputStream with Positioned
}

package is.hail.utils.prettyPrint

import java.io.{ByteArrayInputStream, InputStream}

class ArrayOfByteArrayInputStream(bytes: Array[Array[Byte]]) extends InputStream {

  val byteInputStreams = bytes.map(new ByteArrayInputStream(_))
  var currentInputStreamIdx = 0

  override def read(): Int = {
    var foundByte = false
    var byteToReturn = -1
    while (!foundByte) {
      if (currentInputStreamIdx == byteInputStreams.length) {
        foundByte = true
      } else {
        val readByte = byteInputStreams(currentInputStreamIdx).read()
        if (readByte == -1) {
          currentInputStreamIdx += 1
        } else {
          foundByte = true
          byteToReturn = readByte
        }
      }
    }
    byteToReturn
  }

  override def read(b: Array[Byte], off: Int, len: Int): Int = {
    var numBytesRead = 0
    var moreToRead = true

    while (numBytesRead < len && moreToRead) {
      if (currentInputStreamIdx == byteInputStreams.length) {
        moreToRead = false
      } else {
        val bytesReadInOneCall =
          byteInputStreams(currentInputStreamIdx).read(b, off + numBytesRead, len - numBytesRead)
        if (bytesReadInOneCall == -1) {
          currentInputStreamIdx += 1
        } else {
          numBytesRead += bytesReadInOneCall
        }
      }
    }

    numBytesRead
  }
}

package is.hail.utils.prettyPrint

import java.io.{ByteArrayInputStream, InputStream}

class ArrayOfByteArrayInputStream(bytes: Array[Array[Byte]]) extends InputStream {

  val byteInputStreams = bytes.map(new ByteArrayInputStream(_))
  var currentInputStreamIdx = 0

  override def read(): Int = {
    var foundByte = false
    var byteToReturn = -1
    while (!foundByte) {
      if (currentInputStreamIdx == bytes.length) {
        foundByte = true
      } else {
        val readByte = byteInputStreams(currentInputStreamIdx).read()
        if (readByte == -1) {
          currentInputStreamIdx += 1
        }
        else {
          foundByte = true
          byteToReturn = readByte
        }
      }
    }
    byteToReturn
  }
}

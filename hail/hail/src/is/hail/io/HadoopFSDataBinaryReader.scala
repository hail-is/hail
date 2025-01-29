package is.hail.io

import is.hail.io.fs.SeekableDataInputStream

class HadoopFSDataBinaryReader(fis: SeekableDataInputStream)
    extends AbstractBinaryReader with AutoCloseable {

  override def read(): Int = fis.read()

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int =
    fis.read(byteArray, hasRead, toRead)

  def close(): Unit = fis.close()

  def seek(pos: Long): Unit = fis.seek(pos)

  def skipBytes(bytes: Long): Long = fis.skip(bytes)

  def getPosition: Long = fis.getPosition
}

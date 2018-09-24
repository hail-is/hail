package is.hail.io

import java.io.InputStream

import org.apache.hadoop.fs.FSDataInputStream

class HadoopFSDataBinaryReader(is: InputStream) extends AbstractBinaryReader {

  val fis = new FSDataInputStream(is)

  override def read(): Int = fis.read()

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int = fis.read(byteArray, hasRead, toRead)

  def close() = fis.close()

  def seek(pos: Long) = fis.seek(pos)

  def skipBytes(bytes: Long): Long = fis.skip(bytes)

  def getPosition: Long = fis.getPos
}
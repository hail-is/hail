package is.hail.io.fs

import java.io._
import java.util

import is.hail.utils._

import scala.io.Source

trait PositionedStream {
  def getPosition: Long
}

trait SeekableStream extends PositionedStream {
  def seek(pos: Long): Unit
}

abstract class SeekableDataInputStream(is: InputStream) extends DataInputStream(is) with SeekableStream

abstract class PositionedDataOutputStream(os: OutputStream) extends DataOutputStream(os) with PositionedStream

trait FileStatus {
  def getPath: String
  def getModificationTime: Long
  def getLen: Long
  def isDirectory: Boolean
  def isFile: Boolean
  def getOwner: String
}

trait FS extends Serializable {
  def getProperty(name: String): String

  def setProperty(name: String, value: String): Unit

  def getProperties: Iterator[util.Map.Entry[String, String]]

  def openNoCompression(filename: String): SeekableDataInputStream

  def open(filename: String, checkCodec: Boolean = true): InputStream

  def createNoCompression(filename: String): PositionedDataOutputStream

  def create(filename: String): OutputStream

  /**
    * @return true if a new directory was created, false otherwise
    **/
  def mkDir(dirname: String): Boolean

  def delete(filename: String, recursive: Boolean)

  def exists(files: String*): Boolean

  def isFile(filename: String): Boolean

  def isDir(path: String): Boolean

  def listStatus(filename: String): Array[FileStatus]

  def getTemporaryFile(tmpdir: String, nChar: Int = 10,
    prefix: Option[String] = None, suffix: Option[String] = None): String

  def glob(filename: String): Array[FileStatus]

  def globAll(filenames: Iterable[String]): Array[String]

  def globAllStatuses(filenames: Iterable[String]): Array[FileStatus]

  def copy(src: String, dst: String, deleteSource: Boolean = false): Unit

  def copyMerge(sourceFolder: String,
    destinationFile: String,
    numPartFilesExpected: Int,
    deleteSource: Boolean = true,
    header: Boolean = true,
    partFilesOpt: Option[IndexedSeq[String]] = None): Unit

  def copyMergeList(srcFileStatuses: Array[FileStatus], destFilename: String, deleteSource: Boolean = true): Unit

  def stripCodec(s: String): String

  def getCodec(s: String): String

  def fileStatus(filename: String): FileStatus

  def makeQualified(path: String): String

  def deleteOnExit(path: String): Unit

  def getFileSize(filename: String): Long = fileStatus(filename).getLen

  def readLines[T](filename: String, filtAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace())(reader: Iterator[WithContext[String]] => T): T = {
    using(open(filename)) {
      is =>
        val lines = Source.fromInputStream(is)
          .getLines()
          .zipWithIndex
          .map {
            case (value, position) =>
              val source = Context(value, filename, Some(position))
              WithContext(value, source)
          }
        reader(filtAndReplace(lines))
    }
  }

  def writeTable(filename: String, lines: Traversable[String], header: Option[String] = None): Unit = {
    using(new OutputStreamWriter(create(filename))) { fw =>
        header.foreach { h =>
          fw.write(h)
          fw.write('\n')
        }
        lines.foreach { line =>
          fw.write(line)
          fw.write('\n')
        }
    }
  }
}

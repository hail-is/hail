package is.hail.io.fs

import java.io._
import java.util

import is.hail.utils.{TextInputFilterAndReplace, WithContext}
import net.jpountz.lz4.LZ4Compressor
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.hadoop.fs.{ FSDataInputStream, FSDataOutputStream }

trait FileSystem {
  def open: FSDataInputStream
  def open(path: FilePath): FSDataInputStream
  def open(path: String): FSDataInputStream

  def getPath(path: String): FilePath
  def makeQualified(path: String): FilePath
  def makeQualified(path: FilePath): FilePath
  def deleteOnExit(path: FilePath): Boolean
}

trait FileStatus {
  def getPath: FilePath
  def getModificationTime: Long
  def getLen: Long
  def isDirectory: Boolean
  def isFile: Boolean
  def getOwner: String
}

trait FilePath extends Serializable{
  type Configuration

  def toString: String
  def getName: String
  def getFileSystem(conf: Configuration): FileSystem
}

trait FS extends Serializable{
  def getProperty(name: String): String

  def setProperty(name: String, value: String): Unit

  def getProperties: Iterator[util.Map.Entry[String, String]]

  protected def open(filename: String, checkCodec: Boolean = true): InputStream
  /**
    * @return true if a new directory was created, false otherwise
    **/
  def mkDir(dirname: String): Boolean

  def delete(filename: String, recursive: Boolean)

  def exists(files: String*): Boolean

  def isFile(filename: String): Boolean

  def isDir(path: String): Boolean

  def listStatus(filename: String): Array[FileStatus]

  def fileSystem(filename: String): FileSystem

  def getFileSize(filename: String): Long

  def getTemporaryFile(tmpdir: String, nChar: Int = 10,
                       prefix: Option[String] = None, suffix: Option[String] = None): String

  def glob(filename: String): Array[FileStatus]

  def globAll(filenames: Iterable[String]): Array[String]

  def globAllStatuses(filenames: Iterable[String]): Array[FileStatus]

  def copy(src: String, dst: String, deleteSource: Boolean = false): Unit

  def copyMerge( sourceFolder: String,
                 destinationFile: String,
                 numPartFilesExpected: Int,
                 deleteSource: Boolean = true,
                 header: Boolean = true,
                 partFilesOpt: Option[IndexedSeq[String]] = None ): Unit

  def copyMergeList(srcFileStatuses: Array[FileStatus], destFilename: String, deleteSource: Boolean = true)

  def stripCodec(s: String): String

  def getCodec(s: String): String

  def fileStatus(filename: String): FileStatus

  def writeObjectFile[T](filename: String)(f: (ObjectOutputStream) => T): T

  def readObjectFile[T](filename: String)(f: (ObjectInputStream) => T): T

  def writeDataFile[T](filename: String)(f: (DataOutputStream) => T): T

  def readDataFile[T](filename: String)(f: (DataInputStream) => T): T

  def writeTextFile[T](filename: String)(f: (OutputStreamWriter) => T): T

  def readTextFile[T](filename: String)(f: (InputStreamReader) => T): T

  def writeKryoFile[T](filename: String)(f: (Output) => T): T

  def readKryoFile[T](filename: String)(f: (Input) => T): T

  def readFile[T](filename: String)(f: (InputStream) => T): T

  def writeFile[T](filename: String)(f: (OutputStream) => T): T

  def readFileNoCompression[T](filename: String)(f: (FSDataInputStream) => T): T

  def writeFileNoCompression[T](filename: String)(f: (FSDataOutputStream) => T): T

  def readLines[T](filename: String, filtAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace())(reader: Iterator[WithContext[String]] => T): T

  def writeTable(filename: String, lines: Traversable[String], header: Option[String] = None)

  def writeLZ4DataFile[T](path: String, blockSize: Int, compressor: LZ4Compressor)(writer: (DataOutputStream) => T): T

  def unsafeReader(filename: String, checkCodec: Boolean = true): InputStream

  def unsafeWriter(filename: String): OutputStream
}


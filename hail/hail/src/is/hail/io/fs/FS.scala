package is.hail.io.fs

import is.hail.io.compress.{BGzipInputStream, BGzipOutputStream}
import is.hail.io.fs.FSUtil.{containsWildcard, dropTrailingSlash}
import is.hail.utils._

import scala.collection.mutable

import java.io._
import java.nio.ByteBuffer
import java.nio.file.FileSystems
import java.util.zip.GZIPOutputStream

import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.IOUtils

class WrappedSeekableDataInputStream(is: SeekableInputStream)
    extends DataInputStream(is) with Seekable {
  override def getPosition: Long = is.getPosition

  override def seek(pos: Long): Unit = is.seek(pos)
}

class WrappedPositionedDataOutputStream(os: PositionedOutputStream)
    extends DataOutputStream(os) with Positioned {
  override def getPosition: Long = os.getPosition
}

class WrappedPositionOutputStream(os: OutputStream) extends OutputStream with Positioned {
  private[this] var count: Long = 0L

  override def flush(): Unit = os.flush()

  override def write(i: Int): Unit = {
    os.write(i)
    count += 1
  }

  override def write(bytes: Array[Byte], off: Int, len: Int): Unit =
    os.write(bytes, off, len)

  override def close(): Unit =
    os.close()

  override def getPosition: Long = count
}

abstract class FSURL[URL <: FSURL[URL]] {
  def path: String
  def /(component: String): URL
}

trait FileStatus {
  def getPath: String
  def getActualUrl: String
  def getModificationTime: java.lang.Long
  def getLen: Long
  def isSymlink: Boolean
  def getOwner: String
  def isFileOrFileAndDirectory: Boolean = true
}

trait FileListEntry extends FileStatus {
  def isFile: Boolean
  def isDirectory: Boolean
  override def isFileOrFileAndDirectory: Boolean = isFile
}

class BlobStorageFileStatus(
  actualUrl: String,
  modificationTime: java.lang.Long,
  size: Long,
) extends FileStatus {
  // NB: it is called getPath but it *must* return the URL *with* the scheme.
  override def getPath: String =
    dropTrailingSlash(
      actualUrl
    ) // getPath is a backwards compatible method: in the past, Hail dropped trailing slashes
  override def getActualUrl: String = actualUrl
  override def getModificationTime: java.lang.Long = modificationTime
  override def getLen: Long = size
  override def isSymlink: Boolean = false
  override def getOwner: String = null
}

class BlobStorageFileListEntry(
  actualUrl: String,
  modificationTime: java.lang.Long,
  size: Long,
  isDir: Boolean,
) extends BlobStorageFileStatus(
      actualUrl,
      modificationTime,
      size,
    ) with FileListEntry {
  override def isDirectory: Boolean = isDir
  override def isFile: Boolean = !isDir
  override def isFileOrFileAndDirectory = isFile
  override def toString: String = s"BSFLE($actualUrl $modificationTime $size $isDir)"

}

trait CompressionCodec {
  def makeInputStream(is: InputStream): InputStream

  def makeOutputStream(os: OutputStream): OutputStream
}

object GZipCompressionCodec extends CompressionCodec {
  // java.util.zip.GZIPInputStream does not support concatenated files/multiple blocks
  override def makeInputStream(is: InputStream): InputStream =
    new GzipCompressorInputStream(is, true)

  override def makeOutputStream(os: OutputStream): OutputStream = new GZIPOutputStream(os)
}

object BGZipCompressionCodec extends CompressionCodec {
  override def makeInputStream(is: InputStream): InputStream = new BGzipInputStream(is)

  override def makeOutputStream(os: OutputStream): OutputStream = new BGzipOutputStream(os)
}

class FileAndDirectoryException(message: String) extends RuntimeException(message)

object FSUtil {
  def dropTrailingSlash(path: String): String = {
    if (path.isEmpty)
      return path

    if (path.last != '/')
      return path

    var i = path.length - 1
    while (i > 0 && path(i - 1) == '/')
      i -= 1
    path.substring(0, i)
  }

  def containsWildcard(path: String): Boolean = {
    var i = 0
    while (i < path.length) {
      val c = path(i)
      if (c == '\\') {
        i += 1
        if (i < path.length)
          i += 1
        else
          return false
      } else if (c == '*' || c == '{' || c == '?' || c == '[')
        return true

      i += 1
    }

    false
  }
}

abstract class FSSeekableInputStream extends InputStream with Seekable {
  protected[this] var closed: Boolean = false
  private[this] var pos: Long = 0
  private[this] var eof: Boolean = false

  protected[this] val bb: ByteBuffer = ByteBuffer.allocate(8 * 1024 * 1024)
  bb.limit(0)

  def fill(): Int

  override def read(): Int = {
    if (eof)
      return -1

    if (bb.remaining() == 0) {
      val nRead = fill()
      if (nRead == -1) {
        eof = true
        return -1
      }
    }

    pos += 1
    bb.get().toInt & 0xff
  }

  override def read(bytes: Array[Byte], off: Int, len: Int): Int = {
    if (eof)
      return -1

    if (bb.remaining() == 0) {
      val nRead = fill()
      if (nRead == -1) {
        eof = true
        return -1
      }
    }

    val toTransfer = math.min(len, bb.remaining())
    bb.get(bytes, off, toTransfer)
    pos += toTransfer
    toTransfer
  }

  protected def physicalSeek(newPos: Long): Unit

  override def seek(newPos: Long): Unit = {
    eof = false
    val distance = newPos - pos
    val bufferSeekPosition = bb.position() + distance
    if (bufferSeekPosition >= 0 && bufferSeekPosition < bb.limit()) {
      assert(bufferSeekPosition <= Int.MaxValue)
      bb.position(bufferSeekPosition.toInt)
    } else {
      bb.clear()
      bb.limit(0)
      physicalSeek(newPos)
    }
    pos = newPos
  }

  override def getPosition: Long = pos
}

abstract class FSPositionedOutputStream(val capacity: Int) extends OutputStream with Positioned {
  protected[this] var closed: Boolean = false
  protected[this] val bb: ByteBuffer = ByteBuffer.allocate(capacity)
  protected[this] var pos: Long = 0

  override def flush(): Unit

  override def write(i: Int): Unit = {
    if (bb.remaining() == 0)
      flush()
    bb.put(i.toByte)
    pos += 1
  }

  override def write(bytes: Array[Byte], off: Int, len: Int): Unit = {
    var i = off
    var remaining = len
    while (remaining > 0) {
      if (bb.remaining() == 0)
        flush()
      val toTransfer = math.min(bb.remaining(), remaining)
      bb.put(bytes, i, toTransfer)
      i += toTransfer
      remaining -= toTransfer
      pos += toTransfer
    }
  }

  override def getPosition: Long = pos
}

trait FS extends Serializable with Logging {
  type URL <: FSURL[URL]

  def parseUrl(filename: String): URL

  def validUrl(filename: String): Boolean

  final def openNoCompression(filename: String): SeekableDataInputStream =
    openNoCompression(parseUrl(filename))

  def openNoCompression(url: URL): SeekableDataInputStream

  final def createNoCompression(filename: String): PositionedDataOutputStream =
    createNoCompression(parseUrl(filename))

  def createNoCompression(url: URL): PositionedDataOutputStream

  final def mkDir(dirname: String): Unit = mkDir(parseUrl(dirname))

  def mkDir(url: URL): Unit = ()

  final def listDirectory(filename: String): Array[FileListEntry] =
    listDirectory(parseUrl(filename))

  def listDirectory(url: URL): Array[FileListEntry]

  final def delete(filename: String, recursive: Boolean): Unit =
    delete(parseUrl(filename), recursive)

  def delete(url: URL, recursive: Boolean): Unit

  final def glob(filename: String): Array[FileListEntry] = glob(parseUrl(filename))

  def glob(url: URL): Array[FileListEntry]

  def globWithPrefix(prefix: URL, path: String): Array[FileListEntry] = {
    val components =
      if (path == "")
        Array.empty[String]
      else
        path.split("/")

    val javaFS = FileSystems.getDefault

    val ab = new mutable.ArrayBuffer[FileListEntry]()
    def f(prefix: URL, fs: FileListEntry, i: Int): Unit = {
      if (i == components.length) {
        var t = fs
        if (t == null) {
          try
            t = fileListEntry(prefix)
          catch {
            case _: FileNotFoundException =>
          }
        }
        if (t != null)
          ab += t
      }

      if (i < components.length) {
        val c = components(i)
        if (containsWildcard(c)) {
          val m = javaFS.getPathMatcher(s"glob:$c")
          for (directoryEntry <- listDirectory(prefix)) {
            val p = dropTrailingSlash(directoryEntry.getPath)
            val d = p.drop(prefix.toString.length + 1)
            if (m.matches(javaFS.getPath(d))) {
              f(parseUrl(p), directoryEntry, i + 1)
            }
          }
        } else
          f(prefix / c, null, i + 1)
      }
    }

    f(prefix, null, 0)
    ab.toArray
  }

  def globAll(filenames: Iterable[String]): Array[FileListEntry] =
    filenames.flatMap((x: String) => glob(x)).toArray

  final def eTag(filename: String): Option[String] = eTag(parseUrl(filename))

  /** Return the file's HTTP etag, if the underlying file system supports etags. */
  def eTag(url: URL): Option[String]

  final def fileStatus(filename: String): FileStatus = fileStatus(parseUrl(filename))

  def fileStatus(url: URL): FileStatus

  final def fileListEntry(filename: String): FileListEntry = fileListEntry(parseUrl(filename))

  def fileListEntry(url: URL): FileListEntry

  def makeQualified(path: String): String

  final def open(filename: String, codec: CompressionCodec): InputStream =
    open(parseUrl(filename), codec)

  def open(url: URL, codec: CompressionCodec): InputStream = {
    val is = openNoCompression(url)
    codec.makeInputStream(is)
  }

  final def open(filename: String): InputStream =
    open(parseUrl(filename))

  def open(url: URL): InputStream =
    open(url, gzAsBGZ = false)

  final def open(filename: String, gzAsBGZ: Boolean): InputStream =
    open(parseUrl(filename), gzAsBGZ)

  def open(url: URL, gzAsBGZ: Boolean): InputStream =
    getCodecFromPath(url.path, gzAsBGZ) match {
      case Some(codec) => open(url, codec)
      case _ => openNoCompression(url)
    }

  final def create(filename: String): OutputStream =
    create(parseUrl(filename))

  def create(url: URL): OutputStream =
    getCodecFromPath(url.path) match {
      case Some(codec) => codec.makeOutputStream(createNoCompression(url))
      case None => createNoCompression(url)
    }

  final def write(filename: String)(writer: OutputStream => Unit): Unit =
    write(parseUrl(filename))(writer)

  def write(url: URL)(writer: OutputStream => Unit): Unit =
    using(create(url))(writer)

  final def writePDOS(filename: String)(writer: PositionedDataOutputStream => Unit): Unit =
    writePDOS(parseUrl(filename))(writer)

  def writePDOS(url: URL)(writer: PositionedDataOutputStream => Unit): Unit =
    using(create(url))(os => writer(outputStreamToPositionedDataOutputStream(os)))

  final def getFileSize(filename: String): Long = getFileSize(parseUrl(filename))

  def getFileSize(url: URL): Long = fileStatus(url).getLen

  final def isFile(filename: String): Boolean = isFile(parseUrl(filename))

  final def isFile(url: URL): Boolean =
    try
      fileStatus(url).isFileOrFileAndDirectory
    catch {
      case _: FileNotFoundException => false
    }

  final def isDir(filename: String): Boolean = isDir(parseUrl(filename))

  final def isDir(url: URL): Boolean =
    try
      fileListEntry(url).isDirectory
    catch {
      case _: FileNotFoundException => false
    }

  final def exists(filename: String): Boolean = exists(parseUrl(filename))

  final def exists(url: URL): Boolean = {
    try {
      fileListEntry(url): Unit
      true
    } catch {
      case _: FileNotFoundException => false
    }
  }

  final def copy(src: String, dst: String): Unit = copy(src, dst, false)

  final def copy(src: String, dst: String, deleteSource: Boolean): Unit =
    copy(parseUrl(src), parseUrl(dst), deleteSource)

  def copy(src: URL, dst: URL, deleteSource: Boolean = false): Unit = {
    using(openNoCompression(src)) { is =>
      using(createNoCompression(dst))(os => IOUtils.copy(is, os): Unit)
    }
    if (deleteSource)
      delete(src, recursive = false)
  }

  final def copyRecode(src: String, dst: String): Unit = copyRecode(src, dst, false)

  final def copyRecode(src: String, dst: String, deleteSource: Boolean): Unit =
    copyRecode(parseUrl(src), parseUrl(dst), deleteSource)

  def copyRecode(src: URL, dst: URL, deleteSource: Boolean = false): Unit = {
    using(open(src))(is => using(create(dst))(os => IOUtils.copy(is, os): Unit))
    if (deleteSource)
      delete(src, recursive = false)
  }

  final def touch(filename: String): Unit = touch(parseUrl(filename))

  def touch(url: URL): Unit =
    using(createNoCompression(url))(_ => ())

  def getConfiguration(): Any

  def setConfiguration(config: Any): Unit
}

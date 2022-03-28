package is.hail.io.fs

import java.io._
import java.util.zip.{GZIPOutputStream}
import is.hail.HailContext
import is.hail.backend.BroadcastValue
import is.hail.io.compress.{BGzipInputStream, BGzipOutputStream}
import is.hail.utils._
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.IOUtils
import org.apache.hadoop

import java.nio.ByteBuffer
import scala.io.Source

trait Positioned {
  def getPosition: Long
}

trait Seekable extends Positioned {
  def seek(pos: Long): Unit
}

class WrappedSeekableDataInputStream(is: SeekableInputStream) extends DataInputStream(is) with Seekable {
  def getPosition: Long = is.getPosition

  def seek(pos: Long): Unit = is.seek(pos)
}

class WrappedPositionedDataOutputStream(os: PositionedOutputStream) extends DataOutputStream(os) with Positioned {
  def getPosition: Long = os.getPosition
}

trait FileStatus {
  def getPath: String
  def getModificationTime: java.lang.Long
  def getLen: Long
  def isDirectory: Boolean
  def isSymlink: Boolean
  def isFile: Boolean
  def getOwner: String
}

class BlobStorageFileStatus(path: String, modificationTime: java.lang.Long, size: Long, isDir: Boolean) extends FileStatus {
  def getPath: String = path
  def getModificationTime: java.lang.Long = modificationTime
  def getLen: Long = size
  def isDirectory: Boolean = isDir
  def isFile: Boolean = !isDir
  def isSymlink: Boolean = false
  def getOwner: String = null
}

trait CompressionCodec {
  def makeInputStream(is: InputStream): InputStream

  def makeOutputStream(os: OutputStream): OutputStream
}

object GZipCompressionCodec extends CompressionCodec {
  // java.util.zip.GZIPInputStream does not support concatenated files/multiple blocks
  def makeInputStream(is: InputStream): InputStream = new GzipCompressorInputStream(is, true)

  def makeOutputStream(os: OutputStream): OutputStream = new GZIPOutputStream(os)
}

object BGZipCompressionCodec extends CompressionCodec {
  def makeInputStream(is: InputStream): InputStream = new BGzipInputStream(is)

  def makeOutputStream(os: OutputStream): OutputStream = new BGzipOutputStream(os)
}

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
}

abstract class FSSeekableInputStream extends InputStream with Seekable {
  protected[this] var closed: Boolean = false
  protected[this] var pos: Long = 0
  protected[this] var eof: Boolean = false

  protected[this] val bb: ByteBuffer = ByteBuffer.allocate(64 * 1024)
  bb.limit(0)

  def fill(): Unit

  override def read(): Int = {
    if (eof)
      return -1

    if (bb.remaining() == 0) {
      fill()
      if (eof)
        return -1
    }

    pos += 1
    bb.get().toInt & 0xff
  }

  override def read(bytes: Array[Byte], off: Int, len: Int): Int = {
    if (eof)
      return -1

    if (bb.remaining() == 0) {
      fill()
      if (eof)
        return -1
    }

    val toTransfer = math.min(len, bb.remaining())
    bb.get(bytes, off, toTransfer)
    pos += toTransfer
    toTransfer
  }

  def getPosition: Long = pos
}

abstract class FSPositionedOutputStream extends OutputStream with Positioned {
  protected[this] var closed: Boolean = false
  protected[this] val bb: ByteBuffer = ByteBuffer.allocate(64 * 1024)
  protected[this] var pos: Long = 0

   def flush(): Unit

   def write(i: Int): Unit = {
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

  def getPosition: Long = pos
}

trait FS extends Serializable {
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

  def getCodecFromExtension(extension: String, gzAsBGZ: Boolean = false): CompressionCodec = {
    extension match {
      case ".gz" =>
        if (gzAsBGZ)
          BGZipCompressionCodec
        else
          GZipCompressionCodec
      case ".bgz" =>
        BGZipCompressionCodec
      case ".tbi" =>
        BGZipCompressionCodec
      case _ =>
        null
    }
  }

  def getCodecFromPath(path: String, gzAsBGZ: Boolean = false): CompressionCodec =
    getCodecFromExtension(getExtension(path), gzAsBGZ)

  def getExtension(path: String): String = {
    var i = path.length - 1
    while (i >= 0) {
      if (i == 0)
        return ""

      val c = path(i)
      if (c == '.') {
        if (path(i - 1) == '/')
          return ""
        else
          return path.substring(i)
      }
      if (c == '/')
        return ""
      i -= 1
    }

    throw new AssertionError("unreachable")
  }

  def stripCodecExtension(path: String): String = {
    val ext = getCodecExtension(path)
    path.dropRight(ext.length)
  }

  def getCodecExtension(path: String): String = {
    val ext = getExtension(path)
    if (ext == ".gz" || ext == ".bgz" || ext == ".tbi")
      ext
    else
      ""
  }

  def openNoCompression(filename: String): SeekableDataInputStream

  def createNoCompression(filename: String): PositionedDataOutputStream

  def mkDir(dirname: String): Unit = ()

  def delete(filename: String, recursive: Boolean)

  def listStatus(filename: String): Array[FileStatus]

  def glob(filename: String): Array[FileStatus]

  def globAll(filenames: Iterable[String]): Array[String] =
    globAllStatuses(filenames).map(_.getPath)

  def globAllStatuses(filenames: Iterable[String]): Array[FileStatus] = filenames.flatMap(glob).toArray

  def fileStatus(filename: String): FileStatus

  def makeQualified(path: String): String

  def deleteOnExit(filename: String): Unit = {
    Runtime.getRuntime.addShutdownHook(
      new Thread(() => delete(filename, recursive = false)))
  }

  def open(path: String, codec: CompressionCodec): InputStream = {
    val is = openNoCompression(path)
    if (codec != null)
      codec.makeInputStream(is)
    else
      is

  }

  def open(path: String): InputStream =
    open(path, gzAsBGZ = false)

  def open(path: String, gzAsBGZ: Boolean): InputStream =
    open(path, getCodecFromPath(path, gzAsBGZ))

  def create(path: String): OutputStream = {
    val os = createNoCompression(path)

    val codec = getCodecFromPath(path, gzAsBGZ = false)
    if (codec != null)
      codec.makeOutputStream(os)
    else
      os
  }

  def getFileSize(filename: String): Long = fileStatus(filename).getLen

  def isFile(filename: String): Boolean = {
    try {
      fileStatus(filename).isFile
    } catch {
      case _: FileNotFoundException => false
    }
  }

  def isDir(filename: String): Boolean = {
    try {
      fileStatus(filename).isDirectory
    } catch {
      case _: FileNotFoundException => false
    }
  }

  def exists(filename: String): Boolean = {
    try {
      fileStatus(filename)
      true
    } catch {
      case _: FileNotFoundException => false
    }
  }

  def copy(src: String, dst: String, deleteSource: Boolean = false) {
    using(openNoCompression(src)) { is =>
      using(createNoCompression(dst)) { os =>
        IOUtils.copy(is, os)
      }
    }
    if (deleteSource)
      delete(src, recursive = false)
  }

  def copyRecode(src: String, dst: String, deleteSource: Boolean = false) {
    using(open(src)) { is =>
      using(create(dst)) { os =>
        IOUtils.copy(is, os)
      }
    }
    if (deleteSource)
      delete(src, recursive = false)
  }

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

  def copyMerge(
    sourceFolder: String,
    destinationFile: String,
    numPartFilesExpected: Int,
    deleteSource: Boolean = true,
    header: Boolean = true,
    partFilesOpt: Option[IndexedSeq[String]] = None
  ) {
    if (!exists(sourceFolder + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    delete(destinationFile, recursive = true) // overwriting by default

    val headerFileStatus = glob(sourceFolder + "/header")

    if (header && headerFileStatus.isEmpty)
      fatal(s"Missing header file")
    else if (!header && headerFileStatus.nonEmpty)
      fatal(s"Found unexpected header file")

    val partFileStatuses = partFilesOpt match {
      case None => glob(sourceFolder + "/part-*")
      case Some(files) => files.map(f => fileStatus(sourceFolder + "/" + f)).toArray
    }
    val sortedPartFileStatuses = partFileStatuses.sortBy(fs => getPartNumber(new hadoop.fs.Path(fs.getPath).getName))
    if (sortedPartFileStatuses.length != numPartFilesExpected)
      fatal(s"Expected $numPartFilesExpected part files but found ${ sortedPartFileStatuses.length }")

    val filesToMerge = headerFileStatus ++ sortedPartFileStatuses

    info(s"merging ${ filesToMerge.length } files totalling " +
      s"${ readableBytes(sortedPartFileStatuses.map(_.getLen).sum) }...")

    val (_, dt) = time {
      copyMergeList(filesToMerge, destinationFile, deleteSource)
    }

    info(s"while writing:\n    $destinationFile\n  merge time: ${ formatTime(dt) }")

    if (deleteSource) {
      delete(sourceFolder, recursive = true)
      if (header)
        delete(sourceFolder + ".header", recursive = false)
    }
  }

  def copyMergeList(srcFileStatuses: Array[FileStatus], destFilename: String, deleteSource: Boolean = true) {
    val codec = Option(getCodecFromPath(destFilename))
    val isBGzip = codec.exists(_ == BGZipCompressionCodec)

    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destFilename && fileStatus.isFile
    })

    using(createNoCompression(destFilename)) { os =>

      var i = 0
      while (i < srcFileStatuses.length) {
        val fileStatus = srcFileStatuses(i)
        val lenAdjust: Long = if (isBGzip && i < srcFileStatuses.length - 1)
          -28
        else
          0
        using(openNoCompression(fileStatus.getPath)) { is =>
          hadoop.io.IOUtils.copyBytes(is, os,
            fileStatus.getLen + lenAdjust,
            false)
        }
        i += 1
      }
    }

    if (deleteSource) {
      srcFileStatuses.foreach { fileStatus =>
        delete(fileStatus.getPath.toString, recursive = true)
      }
    }
  }

  def concatenateFiles(sourceNames: Array[String], destFilename: String): Unit = {
    val fileStatuses = sourceNames.map(fileStatus(_))

    info(s"merging ${ fileStatuses.length } files totalling " +
      s"${ readableBytes(fileStatuses.map(_.getLen).sum) }...")

    val (_, timing) = time(copyMergeList(fileStatuses, destFilename, deleteSource = false))

    info(s"while writing:\n    $destFilename\n  merge time: ${ formatTime(timing) }")
  }

  def touch(filename: String): Unit = {
    using(createNoCompression(filename))(_ => ())
  }

  lazy val broadcast: BroadcastValue[FS] = HailContext.backend.broadcast(this)
}

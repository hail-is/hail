package is.hail.io.fs

import is.hail.backend.BroadcastValue
import is.hail.io.compress.{BGzipInputStream, BGzipOutputStream}
import is.hail.io.fs.FSUtil.{containsWildcard, dropTrailingSlash}
import is.hail.services._
import is.hail.utils._
import is.hail.{HailContext, HailFeatureFlags}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.IOUtils
import org.apache.hadoop

import java.io._
import java.nio.ByteBuffer
import java.nio.charset._
import java.nio.file.FileSystems
import java.util.zip.GZIPOutputStream
import scala.collection.mutable
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

class WrappedPositionOutputStream(os: OutputStream) extends OutputStream with Positioned {
  private[this] var count: Long = 0L

  override def flush(): Unit = os.flush()

  override def write(i: Int): Unit = {
    os.write(i)
    count += 1
  }

  override def write(bytes: Array[Byte], off: Int, len: Int): Unit = {
    os.write(bytes, off, len)
  }

  override def close(): Unit = {
    os.close()
  }

  def getPosition: Long = count
}

trait FSURL[T <: FSURL[T]] {
  def getPath: String
  def addPathComponent(component: String): T
  def fromString(s: String): T

  override def toString(): String
}

trait FileStatus {
  def getPath: String
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
  path: String, modificationTime: java.lang.Long, size: Long
) extends FileStatus {
  def getPath: String = path
  def getModificationTime: java.lang.Long = modificationTime
  def getLen: Long = size
  def isSymlink: Boolean = false
  def getOwner: String = null
}

class BlobStorageFileListEntry(
  path: String, modificationTime: java.lang.Long, size: Long, isDir: Boolean
) extends BlobStorageFileStatus(
  path, modificationTime, size
) with FileListEntry {
  def isDirectory: Boolean = isDir
  def isFile: Boolean = !isDir
  override def isFileOrFileAndDirectory = isFile
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

  def seek(newPos: Long): Unit = {
    eof = false
    val distance = newPos - pos
    val bufferSeekPosition = bb.position() + distance
    if (bufferSeekPosition >= 0 && bufferSeekPosition < bb.limit()) {
      assert(bufferSeekPosition <= Int.MaxValue)
      bb.position(bufferSeekPosition.toInt)
    } else {
      bb.clear()
      bb.limit(0)
      if (bb.remaining() != 0) {
        assert(false, bb.remaining().toString())
      }
      physicalSeek(newPos)
    }
    pos = newPos
  }

  def getPosition: Long = pos
}

abstract class FSPositionedOutputStream(val capacity: Int) extends OutputStream with Positioned {
  protected[this] var closed: Boolean = false
  protected[this] val bb: ByteBuffer = ByteBuffer.allocate(capacity)
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

object FS {
  def cloudSpecificFS(
    credentialsPath: String,
    flags: Option[HailFeatureFlags]
  ): FS = retryTransientErrors {
    val cloudSpecificFS = using(new FileInputStream(credentialsPath)) { is =>
      val credentialsStr = Some(IOUtils.toString(is, Charset.defaultCharset()))
      sys.env.get("HAIL_CLOUD") match {
        case Some("gcp") =>
          val requesterPaysConfiguration = flags.flatMap { flags =>
            RequesterPaysConfiguration.fromFlags(
              flags.get("gcs_requester_pays_project"), flags.get("gcs_requester_pays_buckets")
            )
          }
          new GoogleStorageFS(credentialsStr, requesterPaysConfiguration)
        case Some("azure") =>
          new AzureStorageFS(credentialsStr)
        case Some(cloud) =>
          throw new IllegalArgumentException(s"Bad cloud: $cloud")
        case None =>
          throw new IllegalArgumentException(s"HAIL_CLOUD must be set.")
      }
    }

    new RouterFS(Array(cloudSpecificFS, new HadoopFS(new SerializableHadoopConfiguration(new hadoop.conf.Configuration()))))
  }

  def fileListEntryFromIterator[T <: FSURL[T]](
    url: T,
    it: Iterator[FileListEntry],
    makeDirFle: String => FileListEntry
  ): FileListEntry = {
    if (url.getPath == "")
      return makeDirFle(url.toString)

    val prefix = dropTrailingSlash(url.toString)
    val prefixWithSlash = prefix + "/"

    var continue = it.hasNext
    var fileFle: FileListEntry = null
    var dirFle: FileListEntry = null
    while (continue) {
      val fle = it.next()

      if (fle.getPath == prefix) {
        assert(fle.isFile)
        fileFle = fle
      }

      if (fle.getPath == prefixWithSlash) {
        assert(fle.isDirectory)
        dirFle = fle
      }

      continue = it.hasNext && (fle.getPath <= prefixWithSlash)
    }

    if (fileFle != null) {
      if (dirFle != null) {
        throw new FileAndDirectoryException(prefix)
      } else {
        fileFle
      }
    } else {
      if (dirFle != null) {
        dirFle
      } else {
        throw new FileNotFoundException(url.toString)
      }
    }
  }
}

trait FS extends Serializable {
  type URL <: FSURL[URL]

  def validUrl(filename: String): Boolean

  def openCachedNoCompression(filename: String): SeekableDataInputStream = openNoCompression(filename)

  def createCachedNoCompression(filename: String): PositionedDataOutputStream = createNoCompression(filename)

  def writeCached(filename: String)(writer: PositionedDataOutputStream => Unit) = writePDOS(filename)(writer)

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

  final def openNoCompression(filename: String): SeekableDataInputStream = openNoCompression(filename, false)
  def openNoCompression(filename: String, _debug: Boolean): SeekableDataInputStream

  def readNoCompression(filename: String): Array[Byte] = retryTransientErrors {
    using(openNoCompression(filename)) { is =>
      IOUtils.toByteArray(is)
    }
  }

  def createNoCompression(filename: String): PositionedDataOutputStream

  def mkDir(dirname: String): Unit = ()

  def delete(filename: String, recursive: Boolean)

  def listDirectory(filename: String): Array[FileListEntry]

  def listDirectory(url: URL): Array[FileListEntry] = listDirectory(url.toString)

  def glob(filename: String): Array[FileListEntry]

  def globWithPrefix(prefix: URL, path: String): Array[FileListEntry] = {
    val components =
      if (path == "")
        Array.empty[String]
      else
        path.split("/")

    val javaFS = FileSystems.getDefault

    val ab = new mutable.ArrayBuffer[FileListEntry]()
    def f(prefix: URL, fle: FileListEntry, i: Int): Unit = {
      assert(!prefix.getPath.endsWith("/"), prefix)

      if (i == components.length) {
        var t = fle
        if (t == null) {
          try {
            t = getFileListEntry(prefix)
          } catch {
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
          for (cfs <- listDirectory(prefix)) {
            val p = dropTrailingSlash(cfs.getPath)
            val d = p.drop(prefix.toString.length + 1)
            if (m.matches(javaFS.getPath(d))) {
              f(prefix.fromString(p), cfs, i + 1)
            }
          }
        } else
          f(prefix.addPathComponent(c), null, i + 1)
      }
    }

    f(prefix, null, 0)
    ab.toArray
  }

  def globAll(filenames: Iterable[String]): Array[FileListEntry] = filenames.flatMap(glob).toArray

  def fileStatus(filename: String): FileStatus

  def fileStatus(url: URL): FileStatus

  def getFileListEntry(filename: String): FileListEntry

  def getFileListEntry(url: URL): FileListEntry = getFileListEntry(url.toString)

  def makeQualified(path: String): String

  def deleteOnExit(filename: String): Unit = {
    Runtime.getRuntime.addShutdownHook(
      new Thread(() => delete(filename, recursive = false)))
  }

  def open(path: String, codec: CompressionCodec, _debug: Boolean = false): InputStream = {
    val is = openNoCompression(path, _debug)
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

  def write(filename: String)(writer: OutputStream => Unit) =
    using(create(filename))(writer)

  def writePDOS(filename: String)(writer: PositionedDataOutputStream => Unit) =
    using(create(filename))(os => writer(outputStreamToPositionedDataOutputStream(os)))

  def getFileSize(filename: String): Long = fileStatus(filename).getLen

  def isFile(filename: String): Boolean = {
    try {
      getFileListEntry(filename).isFile
    } catch {
      case _: FileNotFoundException => false
    }
  }

  def isDir(filename: String): Boolean = {
    try {
      getFileListEntry(filename).isDirectory
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

    val headerFLEs = glob(sourceFolder + "/header")

    if (header && headerFLEs.isEmpty)
      fatal(s"Missing header file")
    else if (!header && headerFLEs.nonEmpty)
      fatal(s"Found unexpected header file")

    val partFileStatuses = partFilesOpt match {
      case None => glob(sourceFolder + "/part-*")
      case Some(files) => files.map(f => fileStatus(sourceFolder + "/" + f)).toArray
    }

    val sortedPartFileStatuses = partFileStatuses.sortBy { fileStatus =>
      getPartNumber(fileStatus.getPath)
    }

    if (sortedPartFileStatuses.length != numPartFilesExpected)
      fatal(s"Expected $numPartFilesExpected part files but found ${ sortedPartFileStatuses.length }")

    val filesToMerge: Array[FileStatus] = headerFLEs ++ sortedPartFileStatuses

    info(s"merging ${ filesToMerge.length } files totalling " +
      s"${ readableBytes(filesToMerge.map(_.getLen).sum) }...")

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
      fileStatus => fileStatus.getPath != destFilename && fileStatus.isFileOrFileAndDirectory
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
        delete(fileStatus.getPath, recursive = true)
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

  def getConfiguration(): Any

  def setConfiguration(config: Any): Unit
}

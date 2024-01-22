package is.hail.io.fs

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.backend.BroadcastValue
import is.hail.io.compress.{BGzipInputStream, BGzipOutputStream}
import is.hail.io.fs.FSUtil.{containsWildcard, dropTrailingSlash}
import is.hail.services._
import is.hail.utils._

import scala.collection.mutable
import scala.io.Source

import java.io._
import java.nio.ByteBuffer
import java.nio.charset._
import java.nio.file.FileSystems
import java.util.zip.GZIPOutputStream

import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.IOUtils
import org.apache.hadoop
import org.apache.log4j.Logger

class WrappedSeekableDataInputStream(is: SeekableInputStream)
    extends DataInputStream(is) with Seekable {
  def getPosition: Long = is.getPosition

  def seek(pos: Long): Unit = is.seek(pos)
}

class WrappedPositionedDataOutputStream(os: PositionedOutputStream)
    extends DataOutputStream(os) with Positioned {
  def getPosition: Long = os.getPosition
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

  def getPosition: Long = count
}

trait FSURL {
  def getPath: String
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
  def getPath: String =
    dropTrailingSlash(
      actualUrl
    ) // getPath is a backwards compatible method: in the past, Hail dropped trailing slashes
  def getActualUrl: String = actualUrl
  def getModificationTime: java.lang.Long = modificationTime
  def getLen: Long = size
  def isSymlink: Boolean = false
  def getOwner: String = null
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
  def isDirectory: Boolean = isDir
  def isFile: Boolean = !isDir
  override def isFileOrFileAndDirectory = isFile
  override def toString: String = s"BSFLE($actualUrl $modificationTime $size $isDir)"

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
    flags: Option[HailFeatureFlags],
  ): FS = retryTransientErrors {
    val cloudSpecificFS = using(new FileInputStream(credentialsPath)) { is =>
      val credentialsStr = Some(IOUtils.toString(is, Charset.defaultCharset()))
      sys.env.get("HAIL_CLOUD") match {
        case Some("gcp") =>
          val requesterPaysConfiguration = flags.flatMap { flags =>
            RequesterPaysConfiguration.fromFlags(
              flags.get("gcs_requester_pays_project"),
              flags.get("gcs_requester_pays_buckets"),
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

    new RouterFS(Array(
      cloudSpecificFS,
      new HadoopFS(new SerializableHadoopConfiguration(new hadoop.conf.Configuration())),
    ))
  }

  private val log = Logger.getLogger(getClass.getName())
}

trait FS extends Serializable {
  type URL <: FSURL

  import FS.log

  def parseUrl(filename: String): URL

  def validUrl(filename: String): Boolean

  def urlAddPathComponent(url: URL, component: String): URL

  final def openCachedNoCompression(filename: String): SeekableDataInputStream =
    openNoCompression(filename)

  def openCachedNoCompression(url: URL): SeekableDataInputStream = openNoCompression(url)

  final def createCachedNoCompression(filename: String): PositionedDataOutputStream =
    createNoCompression(filename)

  def createCachedNoCompression(url: URL): PositionedDataOutputStream = createNoCompression(url)

  final def writeCached(filename: String)(writer: PositionedDataOutputStream => Unit) =
    writePDOS(filename)(writer)

  def writeCached(url: URL)(writer: PositionedDataOutputStream => Unit) = writePDOS(url)(writer)

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

  final def openNoCompression(filename: String): SeekableDataInputStream =
    openNoCompression(parseUrl(filename))

  def openNoCompression(url: URL): SeekableDataInputStream

  final def readNoCompression(filename: String): Array[Byte] = readNoCompression(parseUrl(filename))

  def readNoCompression(url: URL): Array[Byte] = retryTransientErrors {
    using(openNoCompression(url))(is => IOUtils.toByteArray(is))
  }

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
          f(urlAddPathComponent(prefix, c), null, i + 1)
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

  protected def fileListEntryFromIterator(
    url: URL,
    it: Iterator[FileListEntry],
  ): FileListEntry = {
    val urlStr = url.toString
    val noSlash = dropTrailingSlash(urlStr)
    val withSlash = noSlash + "/"

    var continue = it.hasNext
    var fileFle: FileListEntry = null
    var trailingSlashFle: FileListEntry = null
    var dirFle: FileListEntry = null
    while (continue) {
      val fle = it.next()

      if (fle.isFile) {
        if (fle.getActualUrl == noSlash) {
          fileFle = fle
        } else if (fle.getActualUrl == withSlash) {
          // This is a *blob* whose name has a trailing slash e.g. "gs://bucket/object/". Users
          // really ought to avoid creating these.
          trailingSlashFle = fle
        }
      } else if (fle.isDirectory && dropTrailingSlash(fle.getActualUrl) == noSlash) {
        // In Google, "directory" entries always have a trailing slash.
        //
        // In Azure, "directory" entries never have a trailing slash.
        dirFle = fle
      }

      continue =
        it.hasNext && (fle.getActualUrl <= withSlash) // cloud storage APIs return blobs in alphabetical order, so we need not keep searching after withSlash
    }

    if (fileFle != null) {
      if (dirFle != null) {
        if (trailingSlashFle != null) {
          throw new FileAndDirectoryException(
            s"${url.toString} appears twice as a file (once with and once without a trailing slash) and once as a directory."
          )
        } else {
          throw new FileAndDirectoryException(
            s"${url.toString} appears as both file ${fileFle.getActualUrl} and directory ${dirFle.getActualUrl}."
          )
        }
      } else {
        if (trailingSlashFle != null) {
          log.warn(
            s"Two blobs exist matching ${url.toString}: once with and once without a trailing slash. We will return the one without a trailing slash."
          )
        }
        fileFle
      }
    } else {
      if (dirFle != null) {
        if (trailingSlashFle != null) {
          log.warn(
            s"A blob with a literal trailing slash exists as well as blobs with that prefix. We will treat this as a directory. ${url.toString}"
          )
        }
        dirFle
      } else {
        if (trailingSlashFle != null) {
          throw new FileNotFoundException(
            s"A blob with a literal trailing slash exists. These are sometimes uses to indicate empty directories. " +
              s"Hail does not support this behavior. This folder is treated as if it does not exist. ${url.toString}"
          )
        } else {
          throw new FileNotFoundException(url.toString)
        }
      }
    }
  }

  final def fileListEntry(filename: String): FileListEntry = fileListEntry(parseUrl(filename))

  def fileListEntry(url: URL): FileListEntry

  def makeQualified(path: String): String

  final def deleteOnExit(filename: String): Unit = deleteOnExit(parseUrl(filename))

  def deleteOnExit(url: URL): Unit =
    Runtime.getRuntime.addShutdownHook(
      new Thread(() => delete(url, recursive = false))
    )

  final def open(filename: String, codec: CompressionCodec): InputStream =
    open(parseUrl(filename), codec)

  def open(url: URL, codec: CompressionCodec): InputStream = {
    val is = openNoCompression(url)
    if (codec != null)
      codec.makeInputStream(is)
    else
      is

  }

  final def open(filename: String): InputStream = open(parseUrl(filename))

  def open(url: URL): InputStream =
    open(url, gzAsBGZ = false)

  final def open(filename: String, gzAsBGZ: Boolean): InputStream =
    open(parseUrl(filename), gzAsBGZ)

  def open(url: URL, gzAsBGZ: Boolean): InputStream =
    open(url, getCodecFromPath(url.getPath, gzAsBGZ))

  final def create(filename: String): OutputStream = create(parseUrl(filename))

  def create(url: URL): OutputStream = {
    val os = createNoCompression(url)

    val codec = getCodecFromPath(url.getPath, gzAsBGZ = false)
    if (codec != null)
      codec.makeOutputStream(os)
    else
      os
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
      fileListEntry(url)
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
      using(createNoCompression(dst))(os => IOUtils.copy(is, os))
    }
    if (deleteSource)
      delete(src, recursive = false)
  }

  final def copyRecode(src: String, dst: String): Unit = copyRecode(src, dst, false)

  final def copyRecode(src: String, dst: String, deleteSource: Boolean): Unit =
    copyRecode(parseUrl(src), parseUrl(dst), deleteSource)

  def copyRecode(src: URL, dst: URL, deleteSource: Boolean = false): Unit = {
    using(open(src))(is => using(create(dst))(os => IOUtils.copy(is, os)))
    if (deleteSource)
      delete(src, recursive = false)
  }

  def readLines[T](
    filename: String,
    filtAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace(),
  )(
    reader: Iterator[WithContext[String]] => T
  ): T = {
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

  def writeTable(filename: String, lines: Traversable[String], header: Option[String] = None)
    : Unit = {
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
    partFilesOpt: Option[IndexedSeq[String]] = None,
  ): Unit = {
    if (!exists(sourceFolder + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    delete(destinationFile, recursive = true) // overwriting by default

    val headerFileListEntry = glob(sourceFolder + "/header")

    if (header && headerFileListEntry.isEmpty)
      fatal(s"Missing header file")
    else if (!header && headerFileListEntry.nonEmpty)
      fatal(s"Found unexpected header file")

    val partFileStatuses: Array[_ <: FileStatus] = partFilesOpt match {
      case None => glob(sourceFolder + "/part-*")
      case Some(files) => files.map(f => fileStatus(sourceFolder + "/" + f)).toArray
    }

    val sortedPartFileStatuses = partFileStatuses.sortBy { fileStatus =>
      getPartNumber(fileStatus.getPath)
    }

    if (sortedPartFileStatuses.length != numPartFilesExpected)
      fatal(s"Expected $numPartFilesExpected part files but found ${sortedPartFileStatuses.length}")

    val filesToMerge: Array[FileStatus] = headerFileListEntry ++ sortedPartFileStatuses

    info(s"merging ${filesToMerge.length} files totalling " +
      s"${readableBytes(filesToMerge.map(_.getLen).sum)}...")

    val (_, dt) = time {
      copyMergeList(filesToMerge, destinationFile, deleteSource)
    }

    info(s"while writing:\n    $destinationFile\n  merge time: ${formatTime(dt)}")

    if (deleteSource) {
      delete(sourceFolder, recursive = true)
      if (header)
        delete(sourceFolder + ".header", recursive = false)
    }
  }

  def copyMergeList(
    srcFileStatuses: Array[_ <: FileStatus],
    destFilename: String,
    deleteSource: Boolean = true,
  ): Unit = {
    val codec = Option(getCodecFromPath(destFilename))
    val isBGzip = codec.exists(_ == BGZipCompressionCodec)

    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destFilename && fileStatus.isFileOrFileAndDirectory
    })

    using(createNoCompression(destFilename)) { os =>
      var i = 0
      while (i < srcFileStatuses.length) {
        val fileListEntry = srcFileStatuses(i)
        val lenAdjust: Long = if (isBGzip && i < srcFileStatuses.length - 1)
          -28
        else
          0
        using(openNoCompression(fileListEntry.getPath)) { is =>
          hadoop.io.IOUtils.copyBytes(is, os, fileListEntry.getLen + lenAdjust, false)
        }
        i += 1
      }
    }

    if (deleteSource) {
      srcFileStatuses.foreach(fileStatus => delete(fileStatus.getPath, recursive = true))
    }
  }

  def concatenateFiles(sourceNames: Array[String], destFilename: String): Unit = {
    val fileStatuses = sourceNames.map(fileStatus(_))

    info(s"merging ${fileStatuses.length} files totalling " +
      s"${readableBytes(fileStatuses.map(_.getLen).sum)}...")

    val (_, timing) = time(copyMergeList(fileStatuses, destFilename, deleteSource = false))

    info(s"while writing:\n    $destFilename\n  merge time: ${formatTime(timing)}")
  }

  final def touch(filename: String): Unit = touch(parseUrl(filename))

  def touch(url: URL): Unit =
    using(createNoCompression(url))(_ => ())

  lazy val broadcast: BroadcastValue[FS] = HailContext.backend.broadcast(this)

  def getConfiguration(): Any

  def setConfiguration(config: Any): Unit
}

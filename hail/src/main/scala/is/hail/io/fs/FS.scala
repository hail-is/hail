package is.hail.io.fs

import java.io._
import java.util.zip.{GZIPInputStream, GZIPOutputStream}

import is.hail.HailContext
import is.hail.backend.BroadcastValue
import is.hail.io.compress.{BGzipInputStream, BGzipOutputStream}
import is.hail.utils._
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.IOUtils
import org.apache.hadoop

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
  def runOnExit(f: Runnable): Unit = {
    Runtime.getRuntime.addShutdownHook(new Thread(f))
  }
}

trait FS extends Serializable {
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

  def mkDir(dirname: String): Unit

  def delete(filename: String, recursive: Boolean)

  def listStatus(filename: String): Array[FileStatus]

  def glob(filename: String): Array[FileStatus]

  def globAll(filenames: Iterable[String]): Array[String]

  def globAllStatuses(filenames: Iterable[String]): Array[FileStatus]

  def fileStatus(filename: String): FileStatus

  def makeQualified(path: String): String

  def deleteOnExit(filename: String): Unit

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

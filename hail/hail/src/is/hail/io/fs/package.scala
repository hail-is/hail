package is.hail.io

import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.utils._

import scala.collection.compat.IterableOnce
import scala.io.Source

import java.io._

import org.apache.hadoop.fs.PathIOException
import org.apache.hadoop.io.IOUtils

package object fs extends Logging {
  type PositionedInputStream = InputStream with Positioned

  type SeekableInputStream = InputStream with Seekable

  type SeekableDataInputStream = DataInputStream with Seekable

  type PositionedOutputStream = OutputStream with Positioned

  type PositionedDataOutputStream = DataOutputStream with Positioned

  def outputStreamToPositionedDataOutputStream(os: OutputStream): PositionedDataOutputStream =
    new WrappedPositionedDataOutputStream(
      new WrappedPositionOutputStream(
        os
      )
    )

  private[this] val PartRegex = """.*/?part-(\d+).*""".r

  def getPartNumber(fname: String): Int =
    fname match {
      case PartRegex(i) => i.toInt
      case _ => throw new PathIOException(s"invalid partition file '$fname'")
    }

  private[this] val Kilo: Long = 1024
  private[this] val Mega: Long = Kilo * 1024
  private[this] val Giga: Long = Mega * 1024
  private[this] val Tera: Long = Giga * 1024

  def readableBytes(bytes: Long): String =
    if (bytes < Kilo) bytes.toString
    else if (bytes < Mega) formatDigits(bytes, Kilo) + "K"
    else if (bytes < Giga) formatDigits(bytes, Mega) + "M"
    else if (bytes < Tera) formatDigits(bytes, Giga) + "G"
    else formatDigits(bytes, Tera) + "T"

  private def formatDigits(n: Long, factor: Long): String =
    "%.1f".format(n / factor.toDouble)

  implicit class FsOps(private val fs: FS) extends AnyVal {
    def readLines[T](
      filename: String,
      filtAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace(),
    )(
      reader: Iterator[WithContext[String]] => T
    ): T = {
      using(fs.open(filename)) { is =>
        val lines = Source.fromInputStream(is)
          .getLines()
          .zipWithIndex
          .map { case (value, position) =>
            val source = Context(value, filename, Some(position))
            WithContext(value, source)
          }
        reader(filtAndReplace(lines))
      }
    }

    def writeTable(filename: String, lines: IterableOnce[String], header: Option[String] = None)
      : Unit =
      using(new OutputStreamWriter(fs.create(filename))) { fw =>
        for (h <- header) {
          fw.write(h)
          fw.write('\n')
        }

        for (line <- lines) {
          fw.write(line)
          fw.write('\n')
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
      if (!fs.exists(sourceFolder + "/_SUCCESS"))
        fatal("write failed: no success indicator found")

      fs.delete(destinationFile, recursive = true) // overwriting by default

      val headerFileListEntry = fs.glob(sourceFolder + "/header")

      if (header && headerFileListEntry.isEmpty)
        fatal(s"Missing header file")
      else if (!header && headerFileListEntry.nonEmpty)
        fatal(s"Found unexpected header file")

      val partFileStatuses: Array[_ <: FileStatus] = partFilesOpt match {
        case None => fs.glob(sourceFolder + "/part-*")
        case Some(files) => files.map(f => fs.fileStatus(sourceFolder + "/" + f)).toArray
      }

      val sortedPartFileStatuses = partFileStatuses.sortBy { fileStatus =>
        getPartNumber(fileStatus.getPath)
      }

      if (sortedPartFileStatuses.length != numPartFilesExpected)
        fatal(
          s"Expected $numPartFilesExpected part files but found ${sortedPartFileStatuses.length}"
        )

      val filesToMerge: Array[FileStatus] = headerFileListEntry ++ sortedPartFileStatuses

      logger.info(s"merging ${filesToMerge.length} files totalling " +
        s"${readableBytes(filesToMerge.map(_.getLen).sum)}...")

      val (_, dt) = time(copyMergeList(filesToMerge, destinationFile, deleteSource))

      logger.info(s"while writing:\n    $destinationFile\n  merge time: ${formatTime(dt)}")

      if (deleteSource) {
        fs.delete(sourceFolder, recursive = true)
        if (header)
          fs.delete(sourceFolder + ".header", recursive = false)
      }
    }

    def copyMergeList(
      srcFileStatuses: Array[_ <: FileStatus],
      destFilename: String,
      deleteSource: Boolean = true,
    ): Unit = {
      val codec = getCodecFromPath(destFilename)
      val isBGzip = codec.contains(BGZipCompressionCodec)

      require(srcFileStatuses.forall {
        fileStatus => fileStatus.getPath != destFilename && fileStatus.isFileOrFileAndDirectory
      })

      using(fs.createNoCompression(destFilename)) { os =>
        for (i <- srcFileStatuses.indices) {
          val fileListEntry = srcFileStatuses(i)
          val lenAdjust: Long =
            if (isBGzip && i < srcFileStatuses.length - 1) -28
            else 0

          using(fs.openNoCompression(fileListEntry.getPath)) { is =>
            IOUtils.copyBytes(is, os, fileListEntry.getLen + lenAdjust, false)
          }
        }
      }

      if (deleteSource) {
        srcFileStatuses.foreach(fileStatus => fs.delete(fileStatus.getPath, recursive = true))
      }
    }

    def concatenateFiles(sourceNames: Array[String], destFilename: String): Unit = {
      val fileStatuses = sourceNames.map(fs.fileStatus)

      logger.info(s"merging ${fileStatuses.length} files totalling " +
        s"${readableBytes(fileStatuses.map(_.getLen).sum)}...")

      val (_, timing) = time(fs.copyMergeList(fileStatuses, destFilename, deleteSource = false))

      logger.info(s"while writing:\n    $destFilename\n  merge time: ${formatTime(timing)}")
    }
  }

  def getCodecFromExtension(extension: String, gzAsBGZ: Boolean = false): Option[CompressionCodec] =
    extension match {
      case ".gz" => Some(if (gzAsBGZ) BGZipCompressionCodec else GZipCompressionCodec)
      case ".bgz" => Some(BGZipCompressionCodec)
      case ".tbi" => Some(BGZipCompressionCodec)
      case _ => None
    }

  def getCodecFromPath(path: String, gzAsBGZ: Boolean = false): Option[CompressionCodec] =
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

  def getCodecExtension(path: String): String = {
    val ext = getExtension(path)
    if (ext == ".gz" || ext == ".bgz" || ext == ".tbi")
      ext
    else
      ""
  }

  private[fs] def fileListEntryFromIterator(
    url: FSURL[_],
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
          logger.warn(
            s"Two blobs exist matching ${url.toString}: once with and once without a trailing slash. We will return the one without a trailing slash."
          )
        }
        fileFle
      }
    } else {
      if (dirFle != null) {
        if (trailingSlashFle != null) {
          logger.warn(
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
}

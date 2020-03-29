package is.hail.io.fs

import is.hail.utils.{fatal, formatTime, getPartNumber, info, time, using, warn, SerializableHadoopConfiguration}
import java.io._
import java.util.Map
import scala.collection.JavaConverters._

import com.esotericsoftware.kryo.io.{Input, Output}
import is.hail.io.compress.BGzipCodec
import is.hail.utils.{Context, TextInputFilterAndReplace, WithContext, readableBytes}
import net.jpountz.lz4.{LZ4BlockOutputStream, LZ4Compressor}
import org.apache.hadoop
import org.apache.hadoop.fs.{ FSDataInputStream, FSDataOutputStream }
import org.apache.hadoop.io.IOUtils._
import org.apache.hadoop.io.compress.CompressionCodecFactory
import is.hail.utils._

import scala.io.Source

class HadoopFileStatus(fs: hadoop.fs.FileStatus) extends FileStatus {
  def getPath: String = fs.getPath.toString

  def getModificationTime: Long = fs.getModificationTime

  def getLen: Long = fs.getLen

  def isDirectory: Boolean = fs.isDirectory

  def isFile: Boolean = fs.isFile

  def getOwner: String = fs.getOwner
}

class HadoopFS(val conf: SerializableHadoopConfiguration) extends FS {
  def create(filename: String): OutputStream = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)

    val os = fs.create(hPath)
    val codecFactory = new CompressionCodecFactory(conf.value)
    val codec = codecFactory.getCodec(hPath)
    if (codec != null)
      codec.createOutputStream(os)
    else
      os
  }

  def createNoCompression(filename: String): OutputStream = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.create(hPath)
  }

  def open(filename: String, checkCodec: Boolean = true): InputStream = {
    val is = openNoCompression(filename)

    if (checkCodec) {
      val codecFactory = new CompressionCodecFactory(conf.value)
      val codec = codecFactory.getCodec(new hadoop.fs.Path(filename))
      if (codec != null)
        codec.createInputStream(is)
      else
        is
    } else
      is
  }

  def openNoCompression(filename: String): FSDataInputStream = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    try {
      fs.open(hPath)
    } catch {
      case e: FileNotFoundException =>
        if (isDir(filename))
          throw new FileNotFoundException(s"'$filename' is a directory (or native Table/MatrixTable)")
        else
          throw e
    }
  }

  def getProperty(name: String): String = {
    conf.value.get(name)
  }

  def setProperty(name: String, value: String): Unit = {
    conf.value.set(name, value)
  }

  def getProperties: Iterator[Map.Entry[String, String]] = {
    conf.value.iterator().asScala
  }

  private def getFileSystem(filename: String): hadoop.fs.FileSystem = {
    new hadoop.fs.Path(filename).getFileSystem(conf.value)
  }

  def getFileSize(filename: String): Long =
    fileStatus(filename).getLen

  def listStatus(filename: String): Array[FileStatus] = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.globStatus(hPath)
      .map(_.getPath)
      .flatMap(fs.listStatus(_))
      .map(new HadoopFileStatus(_))
  }

  def isDir(filename: String): Boolean = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.isDirectory(hPath)
  }

  def isFile(filename: String): Boolean = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.isFile(hPath)
  }

  def exists(files: String*): Boolean = {
    files.forall(filename => getFileSystem(filename).exists(new hadoop.fs.Path(filename)))
  }

  /**
    * @return true if a new directory was created, false otherwise
    **/
  def mkDir(dirname: String): Boolean = {
    getFileSystem(dirname).mkdirs(new hadoop.fs.Path(dirname))
  }

  def delete(filename: String, recursive: Boolean) {
    getFileSystem(filename).delete(new hadoop.fs.Path(filename), recursive)
  }

  def getTemporaryFile(tmpdir: String, nChar: Int = 10,
    prefix: Option[String] = None, suffix: Option[String] = None): String = {

    val destFS = getFileSystem(tmpdir)
    val prefixString = if (prefix.isDefined) prefix.get + "-" else ""
    val suffixString = if (suffix.isDefined) "." + suffix.get else ""

    def getRandomName: String = {
      val randomName = tmpdir + "/" + prefixString + scala.util.Random.alphanumeric.take(nChar).mkString + suffixString
      val fileExists = destFS.exists(new hadoop.fs.Path(randomName))

      if (!fileExists)
        randomName
      else
        getRandomName
    }

    getRandomName
  }

  def globAll(filenames: Iterable[String]): Array[String] = {
    filenames.iterator
      .flatMap { arg =>
        val fss = glob(arg)
        val files = fss.map(_.getPath.toString)
        if (files.isEmpty)
          warn(s"'$arg' refers to no files")
        files
      }.toArray
  }

  def globAllStatuses(filenames: Iterable[String]): Array[FileStatus] = {
    filenames.flatMap { filename =>
      val statuses = glob(filename)
      if (statuses.isEmpty)
        warn(s"'$filename' refers to no files")
      statuses
    }.toArray
  }

  def glob(filename: String): Array[FileStatus] = {
    val fs = getFileSystem(filename)
    val path = new hadoop.fs.Path(filename)

    val files = Option(fs.globStatus(path)).getOrElse(Array())
    log.info(s"globbing path $filename returned ${ files.length } files: ${ files.map(_.getPath.getName).mkString(",") }")
    files.map(fileStatus => new HadoopFileStatus(fileStatus))
  }

  def copy(src: String, dst: String, deleteSource: Boolean = false) {
    hadoop.fs.FileUtil.copy(
      getFileSystem(src), new hadoop.fs.Path(src),
      getFileSystem(dst), new hadoop.fs.Path(dst),
      deleteSource, conf.value)
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
    val destPath = new hadoop.fs.Path(destFilename)

    val codecFactory = new CompressionCodecFactory(conf.value)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(destFilename)))
    val isBGzip = codec.exists(_.isInstanceOf[BGzipCodec])

    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destPath && fileStatus.isFile
    })

    val destFS = destPath.getFileSystem(conf.value)
    val outputStream = destFS.create(destPath)

    try {
      var i = 0
      while (i < srcFileStatuses.length) {
        val fileStatus = srcFileStatuses(i)
        val lenAdjust: Long = if (isBGzip && i < srcFileStatuses.length - 1)
          -28
        else
          0
        val srcPath = new hadoop.fs.Path(fileStatus.getPath)
        val srcFS = srcPath.getFileSystem(conf.value)
        val inputStream = srcFS.open(srcPath)
        try {
          copyBytes(inputStream, outputStream,
            fileStatus.getLen + lenAdjust,
            false)
        } finally {
          inputStream.close()
        }
        i += 1
      }
    } finally {
      outputStream.close()
    }

    if (deleteSource) {
      srcFileStatuses.foreach {
        fileStatus => delete(fileStatus.getPath.toString, recursive = true)
      }
    }
  }

  def stripCodec(s: String): String = {
    val path = new org.apache.hadoop.fs.Path(s)

    Option(new CompressionCodecFactory(conf.value)
      .getCodec(path))
      .map { codec =>
        val ext = codec.getDefaultExtension
        assert(s.endsWith(ext))
        s.dropRight(ext.length)
      }.getOrElse(s)
  }

  def getCodec(s: String): String = {
    val path = new org.apache.hadoop.fs.Path(s)

    Option(new CompressionCodecFactory(conf.value)
      .getCodec(path))
      .map { codec =>
        val ext = codec.getDefaultExtension
        assert(s.endsWith(ext))
        s.takeRight(ext.length)
      }.getOrElse("")
  }

  def fileStatus(filename: String): FileStatus = {
    val p = new hadoop.fs.Path(filename)
    new HadoopFileStatus(p.getFileSystem(conf.value).getFileStatus(p))
  }

  def makeQualified(path: String): String = {
    val ppath = new hadoop.fs.Path(path)
    val pathFS = ppath.getFileSystem(conf.value)
    pathFS.makeQualified(ppath).toString
  }

  def deleteOnExit(path: String): Unit = {
    val ppath = new hadoop.fs.Path(path)
    val pathFS = ppath.getFileSystem(conf.value)
    pathFS.deleteOnExit(ppath)
  }
}

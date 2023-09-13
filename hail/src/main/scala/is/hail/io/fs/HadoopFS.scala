package is.hail.io.fs

import is.hail.utils._
import org.apache.hadoop
import org.apache.hadoop.fs.{EtagSource, FSDataInputStream, FSDataOutputStream}
import org.apache.hadoop.io.MD5Hash

import java.io._
import java.security.MessageDigest
import java.util.Base64
import scala.util.Try

class HadoopFileListEntry(fs: hadoop.fs.FileStatus) extends FileListEntry {
  val normalizedPath = fs.getPath

  def getPath: String = fs.getPath.toString

  def getModificationTime: java.lang.Long = fs.getModificationTime

  def getLen: Long = fs.getLen

  def isDirectory: Boolean = fs.isDirectory

  def isFile: Boolean = fs.isFile

  def isSymlink: Boolean = fs.isSymlink

  def getOwner: String = fs.getOwner
}

object HadoopFS {
  def toPositionedOutputStream(os: FSDataOutputStream): PositionedOutputStream =
    new OutputStream with Positioned {
      private[this] var closed: Boolean = false

      override def write(i: Int): Unit = os.write(i)

      override def write(bytes: Array[Byte], off: Int, len: Int): Unit = os.write(bytes, off, len)

      override def flush(): Unit = if (!closed) os.flush()

      override def close(): Unit = {
        if (!closed) {
          os.close()
          closed = true
        }
      }

      def getPosition: Long = os.getPos
    }

  def toSeekableInputStream(is: FSDataInputStream): SeekableInputStream =
    new InputStream with Seekable {
      private[this] var closed: Boolean = false

      override def read(): Int = is.read()

      override def read(bytes: Array[Byte], off: Int, len: Int): Int = is.read(bytes, off, len)

      override def skip(n: Long): Long = is.skip(n)

      override def close(): Unit = {
        if (!closed) {
          is.close()
          closed = true
        }
      }

      def seek(pos: Long): Unit = is.seek(pos)

      def getPosition: Long = is.getPos
    }
}


case class LocalFSURL(val path: String) extends FSURL[LocalFSURL] {
  def addPathComponent(c: String): LocalFSURL = LocalFSURL(s"$path/$c")
  def getPath: String = path
  def fromString(s: String): LocalFSURL = LocalFSURL(s)
  override def toString(): String = path
}


class HadoopFS(private[this] var conf: SerializableHadoopConfiguration) extends FS {
  type URL = LocalFSURL

  override def validUrl(filename: String): Boolean =
    Try(getFileSystem(filename)).isSuccess

  def getConfiguration(): SerializableHadoopConfiguration = conf

  def setConfiguration(_conf: Any): Unit = {
    conf = _conf.asInstanceOf[SerializableHadoopConfiguration]
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    val os = fs.create(hPath)
    new WrappedPositionedDataOutputStream(
      HadoopFS.toPositionedOutputStream(os))
  }

  def openNoCompression(filename: String): SeekableDataInputStream = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    val is = try {
      fs.open(hPath)
    } catch {
      case e: FileNotFoundException =>
        if (isDir(filename))
          throw new FileNotFoundException(s"'$filename' is a directory (or native Table/MatrixTable)")
        else
          throw e
    }

    new WrappedSeekableDataInputStream(
      HadoopFS.toSeekableInputStream(is))

  }

  def getFileSystem(filename: String): hadoop.fs.FileSystem = {
    new hadoop.fs.Path(filename).getFileSystem(conf.value)
  }

  def listStatus(filename: String): Array[FileListEntry] = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    var statuses = fs.globStatus(hPath)
    if (statuses == null) {
      throw new FileNotFoundException(filename)
    } else {
      statuses.par.map(_.getPath)
        .flatMap(fs.listStatus(_))
        .map(new HadoopFileListEntry(_))
        .toArray
    }
  }

  override def mkDir(dirname: String): Unit = {
    getFileSystem(dirname).mkdirs(new hadoop.fs.Path(dirname))
  }

  def remove(fname: String): Unit = {
    getFileSystem(fname).delete(new hadoop.fs.Path(fname), false)
  }

  def rmtree(dirname: String): Unit = {
    getFileSystem(dirname).delete(new hadoop.fs.Path(dirname), true)
  }

  def delete(filename: String, recursive: Boolean) {
    getFileSystem(filename).delete(new hadoop.fs.Path(filename), recursive)
  }

  override def globAll(filenames: Iterable[String]): Array[String] = {
    filenames.iterator
      .flatMap { arg =>
        val fss = glob(arg)
        val files = fss.map(_.getPath.toString)
        if (files.isEmpty)
          warn(s"'$arg' refers to no files")
        files
      }.toArray
  }

  override def globAllStatuses(filenames: Iterable[String]): Array[FileListEntry] = {
    filenames.flatMap { filename =>
      val statuses = glob(filename)
      if (statuses.isEmpty)
        warn(s"'$filename' refers to no files")
      statuses
    }.toArray
  }

  def glob(filename: String): Array[FileListEntry] = {
    val fs = getFileSystem(filename)
    val path = new hadoop.fs.Path(filename)

    var files = fs.globStatus(path)
    if (files == null)
      files = Array.empty
    log.info(s"globbing path $filename returned ${ files.length } files: ${ files.map(_.getPath.getName).mkString(",") }")
    files.map(fileListEntry => new HadoopFileListEntry(fileListEntry))
  }

  def fileListEntry(filename: String): FileListEntry = {
    val p = new hadoop.fs.Path(filename)
    new HadoopFileListEntry(p.getFileSystem(conf.value).getFileStatus(p))
  }

  override def eTag(filename: String): Option[String] = {
    val p = new hadoop.fs.Path(filename)
    val fs = p.getFileSystem(conf.value)
    if (fs.hasPathCapability(p, "fs.capability.etags.available"))
      Some(fs.getFileStatus(p).asInstanceOf[EtagSource].getEtag)
    else
      None
  }

  def makeQualified(path: String): String = {
    val ppath = new hadoop.fs.Path(path)
    val pathFS = ppath.getFileSystem(conf.value)
    pathFS.makeQualified(ppath).toString
  }

  override def deleteOnExit(filename: String): Unit = {
    val ppath = new hadoop.fs.Path(filename)
    val pathFS = ppath.getFileSystem(conf.value)
    pathFS.deleteOnExit(ppath)
  }

  def supportsScheme(scheme: String): Boolean = {
    if (scheme == "") {
      true
    } else {
      try {
        hadoop.fs.FileSystem.getFileSystemClass(scheme, conf.value)
        true
      } catch {
        case e: hadoop.fs.UnsupportedFileSystemException => false
        case e: Exception => throw e
      }
    }
  }
}

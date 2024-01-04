package is.hail.io.fs

import is.hail.utils._

import java.io._
import scala.util.Try

import org.apache.hadoop
import org.apache.hadoop.fs.{EtagSource, FSDataInputStream, FSDataOutputStream}

class HadoopFileListEntry(fs: hadoop.fs.FileStatus) extends FileListEntry {
  val normalizedPath = fs.getPath

  def getPath: String = fs.getPath.toString

  def getActualUrl: String = fs.getPath.toString

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

      override def close(): Unit =
        if (!closed) {
          os.close()
          closed = true
        }

      def getPosition: Long = os.getPos
    }

  def toSeekableInputStream(is: FSDataInputStream): SeekableInputStream =
    new InputStream with Seekable {
      private[this] var closed: Boolean = false

      override def read(): Int = is.read()

      override def read(bytes: Array[Byte], off: Int, len: Int): Int = is.read(bytes, off, len)

      override def skip(n: Long): Long = is.skip(n)

      override def close(): Unit =
        if (!closed) {
          is.close()
          closed = true
        }

      def seek(pos: Long): Unit = is.seek(pos)

      def getPosition: Long = is.getPos
    }
}

case class HadoopFSURL(path: String, conf: SerializableHadoopConfiguration) extends FSURL {
  private[this] val unqualifiedHadoopPath = new hadoop.fs.Path(path)
  val hadoopFs = unqualifiedHadoopPath.getFileSystem(conf.value)
  val hadoopPath = hadoopFs.makeQualified(unqualifiedHadoopPath)

  def addPathComponent(c: String): HadoopFSURL = HadoopFSURL(s"${hadoopPath.toString}/$c", conf)
  def getPath: String = hadoopPath.toString
  def fromString(s: String): HadoopFSURL = HadoopFSURL(s, conf)
  override def toString(): String = hadoopPath.toString
}

class HadoopFS(private[this] var conf: SerializableHadoopConfiguration) extends FS {
  type URL = HadoopFSURL

  override def parseUrl(filename: String): URL = HadoopFSURL(filename, conf)

  override def validUrl(filename: String): Boolean =
    Try(getFileSystem(filename)).isSuccess

  def urlAddPathComponent(url: URL, component: String): URL = url.addPathComponent(component)

  def getConfiguration(): SerializableHadoopConfiguration = conf

  def setConfiguration(_conf: Any): Unit =
    conf = _conf.asInstanceOf[SerializableHadoopConfiguration]

  def createNoCompression(url: URL): PositionedDataOutputStream = {
    val os = url.hadoopFs.create(url.hadoopPath)
    new WrappedPositionedDataOutputStream(
      HadoopFS.toPositionedOutputStream(os)
    )
  }

  def openNoCompression(url: URL): SeekableDataInputStream = {
    val is =
      try
        url.hadoopFs.open(url.hadoopPath)
      catch {
        case e: FileNotFoundException =>
          if (isDir(url))
            throw new FileNotFoundException(s"'$url' is a directory (or native Table/MatrixTable)")
          else
            throw e
      }

    new WrappedSeekableDataInputStream(
      HadoopFS.toSeekableInputStream(is)
    )

  }

  def getFileSystem(filename: String): hadoop.fs.FileSystem =
    new hadoop.fs.Path(filename).getFileSystem(conf.value)

  def listDirectory(url: URL): Array[FileListEntry] = {
    val statuses = url.hadoopFs.globStatus(url.hadoopPath)
    if (statuses == null) {
      throw new FileNotFoundException(url.toString)
    } else {
      statuses.par.map(_.getPath)
        .flatMap(url.hadoopFs.listStatus(_))
        .map(new HadoopFileListEntry(_))
        .toArray
    }
  }

  override def mkDir(url: URL): Unit =
    url.hadoopFs.mkdirs(url.hadoopPath)

  def remove(fname: String): Unit =
    getFileSystem(fname).delete(new hadoop.fs.Path(fname), false)

  def rmtree(dirname: String): Unit =
    getFileSystem(dirname).delete(new hadoop.fs.Path(dirname), true)

  def delete(url: URL, recursive: Boolean) {
    url.hadoopFs.delete(url.hadoopPath, recursive)
  }

  override def globAll(filenames: Iterable[String]): Array[FileListEntry] = {
    filenames.flatMap { filename =>
      val fles = glob(filename)
      if (fles.isEmpty)
        warn(s"'$filename' refers to no files")
      fles
    }.toArray
  }

  def glob(url: URL): Array[FileListEntry] = {
    var files = url.hadoopFs.globStatus(url.hadoopPath)
    if (files == null)
      files = Array.empty
    log.info(
      s"globbing path $url returned ${files.length} files: ${files.map(_.getPath.getName).mkString(",")}"
    )
    files.map(fileListEntry => new HadoopFileListEntry(fileListEntry))
  }

  override def fileStatus(url: URL): FileStatus = {
    val fle = fileListEntry(url)
    if (fle.isDirectory) {
      throw new FileNotFoundException(url.getPath)
    }
    fle
  }

  def fileListEntry(url: URL): FileListEntry =
    new HadoopFileListEntry(url.hadoopFs.getFileStatus(url.hadoopPath))

  override def eTag(url: URL): Option[String] =
    if (url.hadoopFs.hasPathCapability(url.hadoopPath, "fs.capability.etags.available"))
      Some(url.hadoopFs.getFileStatus(url.hadoopPath).asInstanceOf[EtagSource].getEtag)
    else
      None

  def makeQualified(path: String): String = {
    val ppath = new hadoop.fs.Path(path)
    val pathFS = ppath.getFileSystem(conf.value)
    pathFS.makeQualified(ppath).toString
  }

  override def deleteOnExit(url: URL): Unit =
    url.hadoopFs.deleteOnExit(url.hadoopPath)

  def supportsScheme(scheme: String): Boolean =
    (scheme == "") || {
      try {
        hadoop.fs.FileSystem.getFileSystemClass(scheme, conf.value)
        true
      } catch {
        case _: hadoop.fs.UnsupportedFileSystemException => false
      }
    }
}

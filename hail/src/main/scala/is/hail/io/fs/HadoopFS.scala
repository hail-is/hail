package is.hail.io.fs

import is.hail.utils._

import org.apache.hadoop
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream}

import java.io._

class HadoopFileStatus(fs: hadoop.fs.FileStatus) extends FileStatus {
  val normalizedPath = fs.getPath

  def getPath: String = fs.getPath.toString

  def getModificationTime: java.lang.Long = fs.getModificationTime

  def getLen: Long = fs.getLen

  def isDirectory: Boolean = fs.isDirectory

  def isFile: Boolean = fs.isFile

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

class HadoopFS(val conf: SerializableHadoopConfiguration) extends FS {
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

  def listStatus(filename: String): Array[FileStatus] = {
    val fs = getFileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    var statuses = fs.globStatus(hPath)
    if (statuses == null) {
      throw new FileNotFoundException(filename)
    } else {
      statuses.par.map(_.getPath)
        .flatMap(fs.listStatus(_))
        .map(new HadoopFileStatus(_))
        .toArray
    }
  }

  def mkDir(dirname: String): Unit = {
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

    var files = fs.globStatus(path)
    if (files == null)
      files = Array.empty
    log.info(s"globbing path $filename returned ${ files.length } files: ${ files.map(_.getPath.getName).mkString(",") }")
    files.map(fileStatus => new HadoopFileStatus(fileStatus))
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

  def deleteOnExit(filename: String): Unit = {
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

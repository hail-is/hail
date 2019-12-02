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

class HadoopFilePath(path: hadoop.fs.Path) extends FilePath {
  type Configuration = hadoop.conf.Configuration

  override def toString: String = {
    path.toString()
  }

   def getName: String = {
    path.getName()
  }

  def getFileSystem(conf: Configuration): HadoopFileSystem = {
    new HadoopFileSystem(path.toString, conf)
  }
}

class HadoopFileSystem(val filename: String, conf: hadoop.conf.Configuration) extends FileSystem  {
  private val dPath = new hadoop.fs.Path(filename)
  private val hfs = dPath.getFileSystem(conf)

  def open: FSDataInputStream = {
    val test = hfs.open(dPath)
    hfs.open(dPath)
  }

  def open(fPath: FilePath): FSDataInputStream = {
    hfs.open(new hadoop.fs.Path(fPath.toString()))
  }

  def open(fPath: String): FSDataInputStream = {
    hfs.open(new hadoop.fs.Path(fPath))
  }

  def deleteOnExit(fPath: FilePath): Boolean = {
    hfs.deleteOnExit(new hadoop.fs.Path(fPath.toString()))
  }

  def makeQualified(fPath: FilePath): FilePath = {
    new HadoopFilePath(hfs.makeQualified(new hadoop.fs.Path(fPath.toString())))
  }

  def makeQualified(fPath: String): FilePath = {
    new HadoopFilePath(hfs.makeQualified(new hadoop.fs.Path(fPath.toString())))
  }

  def getPath(path: String): FilePath = {
    new HadoopFilePath(new hadoop.fs.Path(path))
  }
}

class HadoopFileStatus(fs: hadoop.fs.FileStatus) extends FileStatus {
  def getPath: HadoopFilePath = {
    new HadoopFilePath(fs.getPath())
  }

  def getModificationTime: Long = fs.getModificationTime

  def getLen: Long = fs.getLen

  def isDirectory: Boolean = fs.isDirectory

  def isFile: Boolean = fs.isFile

  def getOwner: String = fs.getOwner
}

class HadoopFS(val conf: SerializableHadoopConfiguration) extends FS {
  private def create(filename: String): OutputStream = {
    val fs = _fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)

    val os = fs.create(hPath)
    val codecFactory = new CompressionCodecFactory(conf.value)
    val codec = codecFactory.getCodec(hPath)
    if (codec != null)
      codec.createOutputStream(os)
    else
      os
  }

  private def createNoCompression(filename: String): FSDataOutputStream = {
    val fs = _fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    return fs.create(hPath)
  }

  def open(filename: String, checkCodec: Boolean = true): InputStream = {
    val fs = _fileSystem(filename)
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
    if (checkCodec) {
      val codecFactory = new CompressionCodecFactory(conf.value)
      val codec = codecFactory.getCodec(hPath)
      if (codec != null)
        codec.createInputStream(is)
      else
        is
    } else
      is
  }

  private def openNoCompression(filename: String): FSDataInputStream = {
    val fs = _fileSystem(filename)
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

  private def _fileSystem(filename: String): hadoop.fs.FileSystem = {
    new hadoop.fs.Path(filename).getFileSystem(conf.value)
  }

  def fileSystem(filename: String): HadoopFileSystem = {
    new HadoopFileSystem(filename, conf.value)
  }

  def getFileSize(filename: String): Long =
    fileStatus(filename).getLen

  def listStatus(filename: String): Array[FileStatus] = {
    val fs = _fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.globStatus(hPath)
      .map(_.getPath)
      .flatMap(fs.listStatus(_))
      .map(new HadoopFileStatus(_))
  }

  def isDir(filename: String): Boolean = {
    val fs = _fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.isDirectory(hPath)
  }

  def isFile(filename: String): Boolean = {
    val fs = _fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.isFile(hPath)
  }

  def exists(files: String*): Boolean = {
    files.forall(filename => _fileSystem(filename).exists(new hadoop.fs.Path(filename)))
  }

  /**
    * @return true if a new directory was created, false otherwise
    **/
  def mkDir(dirname: String): Boolean = {
    _fileSystem(dirname).mkdirs(new hadoop.fs.Path(dirname))
  }

  def delete(filename: String, recursive: Boolean) {
    _fileSystem(filename).delete(new hadoop.fs.Path(filename), recursive)
  }

  def getTemporaryFile(tmpdir: String, nChar: Int = 10,
    prefix: Option[String] = None, suffix: Option[String] = None): String = {

    val destFS = _fileSystem(tmpdir)
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
    val fs = _fileSystem(filename)
    val path = new hadoop.fs.Path(filename)

    val files = Option(fs.globStatus(path)).getOrElse(Array())
    log.info(s"globbing path $filename returned ${ files.length } files: ${ files.map(_.getPath.getName).mkString(",") }")
    files.map(fileStatus => new HadoopFileStatus(fileStatus))
  }

  def copy(src: String, dst: String, deleteSource: Boolean = false) {
    hadoop.fs.FileUtil.copy(
      _fileSystem(src), new hadoop.fs.Path(src),
      _fileSystem(dst), new hadoop.fs.Path(dst),
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
    val sortedPartFileStatuses = partFileStatuses.sortBy(fs => getPartNumber(fs.getPath.getName))
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
    val destFS = _fileSystem(destFilename)

    val codecFactory = new CompressionCodecFactory(conf.value)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(destFilename)))
    val isBGzip = codec.exists(_.isInstanceOf[BGzipCodec])

    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destPath && fileStatus.isFile
    })

    val outputStream = destFS.create(destPath)

    try {
      var i = 0
      while (i < srcFileStatuses.length) {
        val fileStatus = srcFileStatuses(i)
        val lenAdjust: Long = if (isBGzip && i < srcFileStatuses.length - 1)
          -28
        else
          0
        val srcFS = fileSystem(fileStatus.getPath.toString)
        val inputStream = srcFS.open(fileStatus.getPath)
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

  def writeObjectFile[T](filename: String)(f: (ObjectOutputStream) => T): T =
    using(create(filename)) { ois => using(new ObjectOutputStream(ois))(f) }

  def readObjectFile[T](filename: String)(f: (ObjectInputStream) => T): T =
    using(open(filename)) { is => using(new ObjectInputStream(is))(f) }

  def writeDataFile[T](filename: String)(f: (DataOutputStream) => T): T =
    using(new DataOutputStream(create(filename)))(f)

  def readDataFile[T](filename: String)(f: (DataInputStream) => T): T =
    using(new DataInputStream(open(filename)))(f)

  def writeTextFile[T](filename: String)(f: (OutputStreamWriter) => T): T =
    using(new OutputStreamWriter(create(filename)))(f)

  def readTextFile[T](filename: String)(f: (InputStreamReader) => T): T =
    using(new InputStreamReader(open(filename)))(f)

  def writeKryoFile[T](filename: String)(f: (Output) => T): T =
    using(new Output(create(filename)))(f)

  def readKryoFile[T](filename: String)(f: (Input) => T): T =
    using(new Input(open(filename)))(f)

  def readFile[T](filename: String)(f: (InputStream) => T): T =
    using(open(filename))(f)

  def writeFile[T](filename: String)(f: (OutputStream) => T): T =
    using(create(filename))(f)

  def readFileNoCompression[T](filename: String)(f: (FSDataInputStream) => T): T =
    using(openNoCompression(filename))(f)

  def writeFileNoCompression[T](filename: String)(f: (FSDataOutputStream) => T): T =
    using(createNoCompression(filename))(f)

  def readLines[T](filename: String, filtAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace())(reader: Iterator[WithContext[String]] => T): T = {
    readFile[T](filename) {
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

  def writeTable(filename: String, lines: Traversable[String], header: Option[String] = None) {
    writeTextFile(filename) {
      fw =>
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

  def unsafeReader(filename: String, checkCodec: Boolean = true): InputStream = open(filename, checkCodec)

  def unsafeWriter(filename: String): OutputStream = create(filename)

  def writeLZ4DataFile[T](path: String, blockSize: Int, compressor: LZ4Compressor)(writer: (DataOutputStream) => T): T = {
    val oos = create(path)
    val comp = new LZ4BlockOutputStream(oos, blockSize, compressor)
    val dos = new DataOutputStream(comp)
    try {
      writer(dos)
    } finally {
      dos.flush()
      dos.close()
    }
  }
}

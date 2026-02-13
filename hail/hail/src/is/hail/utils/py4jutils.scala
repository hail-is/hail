package is.hail.utils

import is.hail.io.fs.{FS, FileListEntry, FileStatus, SeekableDataInputStream}

import scala.jdk.CollectionConverters._

import java.io.{InputStream, OutputStream}

import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.appender.ConsoleAppender
import org.apache.logging.log4j.core.config.builder.api.ConfigurationBuilderFactory
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods

object py4jutils extends Logging {
  def pyHandleException(t: Throwable): (String, String, Int) =
    handleForPython(t)

  def pyIterableToArrayList[T](it: Iterable[T]): java.util.ArrayList[T] = {
    val list = new java.util.ArrayList[T]()
    for (elem <- it)
      list.add(elem)
    list
  }

  // we cannot construct an array because we don't have the class tag
  def pyArrayListToISeq[T](al: java.util.ArrayList[T]): IndexedSeq[T] =
    al.asScala.toIndexedSeq

  def pyLs(fs: FS, path: String): String = {
    val statuses = fs.listDirectory(path)
    JsonMethods.compact(JArray(statuses.map(fs => fileListEntryToJson(fs)).toList))
  }

  def pyFileStatus(fs: FS, path: String): String = {
    val stat = fs.fileStatus(path)
    JsonMethods.compact(fileStatusToJson(stat))
  }

  def pyFileListEntry(fs: FS, path: String): String = {
    val stat = fs.fileListEntry(path)
    JsonMethods.compact(fileListEntryToJson(stat))
  }

  private[this] def fileStatusToJson(fs: FileStatus): JObject =
    JObject(
      "path" -> JString(fs.getPath),
      "size" -> JInt(fs.getLen),
      "is_link" -> JBool(fs.isSymlink),
      "modification_time" ->
        (if (fs.getModificationTime != null)
           JString(
             new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSZ").format(
               new java.util.Date(fs.getModificationTime)
             )
           )
         else
           JNull),
      "owner" -> (
        if (fs.getOwner != null)
          JString(fs.getOwner)
        else
          JNull
      ),
    )

  private[this] def fileListEntryToJson(fs: FileListEntry): JObject =
    JObject(fileStatusToJson(fs).obj :+ ("is_dir" -> JBool(fs.isDirectory)))

  def pyReadFile(fs: FS, path: String, buffSize: Int): HadoopSeekablePyReader =
    new HadoopSeekablePyReader(fs.fileListEntry(path), fs.openNoCompression(path), buffSize)

  def pyReadFileCodec(fs: FS, path: String, buffSize: Int): HadoopPyReader =
    new HadoopPyReader(fs.open(path), buffSize)

  def pyWriteFile(fs: FS, path: String, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && fs.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(fs.createNoCompression(path))
  }

  def pyWriteFileCodec(fs: FS, path: String, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && fs.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(fs.create(path))
  }

  private[this] val LogFormat: String =
    "%d{yyyy-MM-dd HH:mm:ss.SSS} %c{1}: %p: %m%n"

  def pyConfigureLogging(logFile: String, quiet: Boolean, append: Boolean): Unit = {
    val configBuilder =
      ConfigurationBuilderFactory
        .newConfigurationBuilder()
        .setConfigurationName("Hail")

    val layout =
      configBuilder
        .newLayout("PatternLayout")
        .addAttribute("pattern", LogFormat)

    val fileAppender =
      configBuilder
        .newAppender("LOGFILE", "File")
        .addAttribute("append", append.toString)
        .addAttribute("fileName", logFile)
        .add(layout)

    val consoleAppender =
      configBuilder
        .newAppender("CONSOLE", "Console")
        .addAttribute("target", ConsoleAppender.Target.SYSTEM_ERR)
        .add(layout)

    val rootLogger =
      configBuilder
        .newRootLogger(Level.INFO)
        .add(
          configBuilder
            .newAppenderRef(fileAppender.getName)
            .addAttribute("level", Level.INFO)
        )
        .add(
          configBuilder
            .newAppenderRef(consoleAppender.getName)
            .addAttribute("level", if (!quiet) Level.WARN else Level.OFF)
        )

    val sparkLogger =
      configBuilder
        .newLogger("org.apache.spark", Level.WARN)
        .add(configBuilder.newAppenderRef(fileAppender.getName))

    Logging.getLoggerContext.reconfigure(
      configBuilder
        .add(fileAppender)
        .add(consoleAppender)
        .add(rootLogger)
        .add(sparkLogger)
        .build(false)
    )
  }
}

class HadoopPyReader(in: InputStream, buffSize: Int) {
  var eof = false

  val buff = new Array[Byte](buffSize)

  def read(n: Int): Array[Byte] = {
    val bytesRead = in.read(buff, 0, math.min(n, buffSize))
    if (bytesRead < 0)
      new Array[Byte](0)
    else if (bytesRead == n)
      buff
    else
      buff.slice(0, bytesRead)
  }

  def close(): Unit =
    in.close()
}

class HadoopSeekablePyReader(status: FileListEntry, in: SeekableDataInputStream, buffSize: Int)
    extends HadoopPyReader(in, buffSize) {
  def seek(pos: Long, whence: Int): Long = {
    // whence corresponds to python arguments to seek
    // it is validated in python
    // 0 (SEEK_SET) seek to pos
    // 1 (SEEK_CUR) seek to getPosition + pos
    // 2 (SEEK_END) seek to status.getLen + pos

    val new_offset = whence match {
      case 0 => pos
      case 1 => getPosition() + pos
      case 2 => status.getLen + pos
    }

    in.seek(new_offset)
    new_offset
  }

  def getPosition(): Long = in.getPosition
}

class HadoopPyWriter(out: OutputStream) {
  def write(b: Array[Byte]): Unit =
    out.write(b)

  def flush(): Unit =
    out.flush()

  def close(): Unit = {
    out.flush()
    out.close()
  }
}

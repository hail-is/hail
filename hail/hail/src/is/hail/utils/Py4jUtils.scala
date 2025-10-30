package is.hail.utils

import is.hail.expr.JSONAnnotationImpex
import is.hail.io.fs.{FS, FileListEntry, FileStatus, SeekableDataInputStream}
import is.hail.types.virtual.Type

import scala.collection.JavaConverters._

import java.io.{InputStream, OutputStream}
import java.util.Properties

import org.apache.log4j.{LogManager, PropertyConfigurator}
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods

trait Py4jUtils extends Logging {
  def arrayToArrayList[T](arr: Array[T]): java.util.ArrayList[T] = {
    val list = new java.util.ArrayList[T]()
    for (elem <- arr)
      list.add(elem)
    list
  }

  def iterableToArrayList[T](it: Iterable[T]): java.util.ArrayList[T] = {
    val list = new java.util.ArrayList[T]()
    for (elem <- it)
      list.add(elem)
    list
  }

  // we cannot construct an array because we don't have the class tag
  def arrayListToISeq[T](al: java.util.ArrayList[T]): IndexedSeq[T] = al.asScala.toIndexedSeq

  def arrayListToSet[T](al: java.util.ArrayList[T]): Set[T] = al.asScala.toSet

  def javaMapToMap[K, V](jm: java.util.Map[K, V]): Map[K, V] = jm.asScala.toMap

  def makeIndexedSeq[T](arr: Array[T]): IndexedSeq[T] = arr: IndexedSeq[T]

  def makeInt(i: Int): Int = i

  def makeInt(l: Long): Int = l.toInt

  def makeLong(i: Int): Long = i.toLong

  def makeLong(l: Long): Long = l

  def makeFloat(f: Float): Float = f

  def makeFloat(d: Double): Float = d.toFloat

  def makeDouble(f: Float): Double = f.toDouble

  def makeDouble(d: Double): Double = d

  def ls(fs: FS, path: String): String = {
    val statuses = fs.listDirectory(path)
    JsonMethods.compact(JArray(statuses.map(fs => fileListEntryToJson(fs)).toList))
  }

  def fileStatus(fs: FS, path: String): String = {
    val stat = fs.fileStatus(path)
    JsonMethods.compact(fileStatusToJson(stat))
  }

  def fileListEntry(fs: FS, path: String): String = {
    val stat = fs.fileListEntry(path)
    JsonMethods.compact(fileListEntryToJson(stat))
  }

  private def fileStatusToJson(fs: FileStatus): JObject = {
    JObject(
      "path" -> JString(fs.getPath.toString),
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
  }

  private def fileListEntryToJson(fs: FileListEntry): JObject =
    JObject(fileStatusToJson(fs).obj :+ ("is_dir" -> JBool(fs.isDirectory)))

  private val kilo: Long = 1024
  private val mega: Long = kilo * 1024
  private val giga: Long = mega * 1024
  private val tera: Long = giga * 1024

  def readableBytes(bytes: Long): String = {
    if (bytes < kilo)
      bytes.toString
    else if (bytes < mega)
      formatDigits(bytes, kilo) + "K"
    else if (bytes < giga)
      formatDigits(bytes, mega) + "M"
    else if (bytes < tera)
      formatDigits(bytes, giga) + "G"
    else
      formatDigits(bytes, tera) + "T"
  }

  private def formatDigits(n: Long, factor: Long): String =
    "%.1f".format(n / factor.toDouble)

  def readFile(fs: FS, path: String, buffSize: Int): HadoopSeekablePyReader =
    new HadoopSeekablePyReader(fs.fileListEntry(path), fs.openNoCompression(path), buffSize)

  def readFileCodec(fs: FS, path: String, buffSize: Int): HadoopPyReader =
    new HadoopPyReader(fs.open(path), buffSize)

  def writeFile(fs: FS, path: String, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && fs.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(fs.createNoCompression(path))
  }

  def writeFileCodec(fs: FS, path: String, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && fs.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(fs.create(path))
  }

  val LogFormat: String =
    "%d{yyyy-MM-dd HH:mm:ss.SSS} %c{1}: %p: %m%n"

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean): Unit = {
    org.apache.log4j.helpers.LogLog.setInternalDebugging(true)
    org.apache.log4j.helpers.LogLog.setQuietMode(false)
    val logProps = new Properties()

    // uncomment to see log4j LogLog output:
    // logProps.put("log4j.debug", "true")
    logProps.put("log4j.rootLogger", "INFO, logfile")
    logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
    logProps.put("log4j.appender.logfile.append", append.toString)
    logProps.put("log4j.appender.logfile.file", logFile)
    logProps.put("log4j.appender.logfile.threshold", "INFO")
    logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
    logProps.put("log4j.appender.logfile.layout.ConversionPattern", LogFormat)

    if (!quiet) {
      logProps.put("log4j.logger.Hail", "INFO, HailConsoleAppender, HailSocketAppender")
      logProps.put("log4j.appender.HailConsoleAppender", "org.apache.log4j.ConsoleAppender")
      logProps.put("log4j.appender.HailConsoleAppender.target", "System.err")
      logProps.put("log4j.appender.HailConsoleAppender.layout", "org.apache.log4j.PatternLayout")
    } else
      logProps.put("log4j.logger.Hail", "INFO, HailSocketAppender")

    logProps.put("log4j.appender.HailSocketAppender", "is.hail.utils.StringSocketAppender")
    logProps.put("log4j.appender.HailSocketAppender.layout", "org.apache.log4j.PatternLayout")

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)
  }

  def addSocketAppender(hostname: String, port: Int): Unit =
    StringSocketAppender.get().connect(hostname, port, LogFormat)

  def logWarn(msg: String): Unit =
    warn(msg)

  def logInfo(msg: String): Unit =
    info(msg)

  def logError(msg: String): Unit =
    error(msg)

  def makeJSON(t: Type, value: Any): String = {
    val jv = JSONAnnotationImpex.exportAnnotation(value, t)
    JsonMethods.compact(jv)
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

package is.hail.utils

import java.io.{InputStream, OutputStream}

import is.hail.HailContext
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.virtual.Type
import is.hail.io.fs.FileStatus
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._

trait Py4jUtils {
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

  def exists(path: String, hc: HailContext): Boolean = hc.sFS.exists(path)

  def isFile(path: String, hc: HailContext): Boolean = hc.sFS.isFile(path)

  def isDir(path: String, hc: HailContext): Boolean = hc.sFS.isDir(path)

  def ls(path: String, hc: HailContext): String = {
    val statuses = hc.sFS.listStatus(path)
    JsonMethods.compact(JArray(statuses.map(fs => statusToJson(fs)).toList))
  }

  def stat(path: String, hc: HailContext): String = {
    val stat = hc.sFS.fileStatus(path)
    JsonMethods.compact(statusToJson(stat))
  }

  private def statusToJson(fs: FileStatus): JObject = {
    JObject(
      "path" -> JString(fs.getPath.toString),
      "size_bytes" -> JInt(fs.getLen),
      "size" -> JString(readableBytes(fs.getLen)),
      "is_dir" -> JBool(fs.isDirectory),
      "modification_time" -> JString(new java.util.Date(fs.getModificationTime).toString),
      "owner" -> JString(fs.getOwner)
    )
  }

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

  private def formatDigits(n: Long, factor: Long): String = {
    (n / factor.toDouble).formatted("%.1f")
  }

  def readFile(path: String, hc: HailContext, buffSize: Int): HadoopPyReader = hc.sFS.readFile(path) { in =>
    new HadoopPyReader(hc.sFS.unsafeReader(path), buffSize)
  }

  def writeFile(path: String, hc: HailContext, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && hc.sFS.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(hc.sFS.unsafeWriter(path))
  }

  def copyFile(from: String, to: String, hc: HailContext) {
    hc.sFS.copy(from, to)
  }

  def addSocketAppender(hostname: String, port: Int) {
    val app = new StringSocketAppender(hostname, port, HailContext.logFormat)
    consoleLog.addAppender(app)
  }

  def logWarn(msg: String) {
    warn(msg)
  }

  def logInfo(msg: String) {
    info(msg)
  }

  def logError(msg: String) {
    error(msg)
  }

  def fileExists(hc: HailContext, path: String): Boolean = hc.sFS.exists(path) && hc.sFS.isFile(path)

  def dirExists(hc: HailContext, path: String): Boolean = hc.sFS.exists(path) && hc.sFS.isDir(path)

  def mkdir(hc: HailContext, path: String): Boolean = hc.sFS.mkDir(path)

  def copyToTmp(hc: HailContext, path: String, extension: String): String = {
    val codecExt = hc.sFS.getCodec(path)
    val tmpFile = hc.getTemporaryFile(suffix = Some(extension + codecExt))
    hc.sFS.copy(path, tmpFile)
    tmpFile
  }

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

  def close() {
    in.close()
  }
}

class HadoopPyWriter(out: OutputStream) {

  def write(b: Array[Byte]) {
    out.write(b)
  }

  def flush() {
    out.flush()
  }

  def close() {
    out.flush()
    out.close()
  }
}

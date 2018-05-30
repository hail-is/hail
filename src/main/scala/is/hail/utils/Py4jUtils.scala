package is.hail.utils

import java.io.{InputStream, OutputStream}

import is.hail.HailContext
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.Type
import is.hail.table.Table
import is.hail.variant.MatrixTable
import org.apache.hadoop.fs.FileStatus
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

  def exists(path: String, hc: HailContext): Boolean = hc.hadoopConf.exists(path)

  def isFile(path: String, hc: HailContext): Boolean = hc.hadoopConf.isFile(path)

  def isDir(path: String, hc: HailContext): Boolean = hc.hadoopConf.isDir(path)

  def ls(path: String, hc: HailContext): String = {
    val statuses = hc.hadoopConf.listStatus(path)
    JsonMethods.compact(JArray(statuses.map(fs => statusToJson(fs)).toList))
  }

  def stat(path: String, hc: HailContext): String = {
    val stat = hc.hadoopConf.fileStatus(path)
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

  private val kilo = 1024
  private val mega = 1024 * 1024
  private val giga = 1024 * 1024 * 1024
  private val tera = 1024 * 1024 * 1024 * 1024

  private def readableBytes(bytes: Long): String = {
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

  private def formatDigits(n: Long, factor: Int): String = {
    (n / factor.toDouble).formatted("%.1f")
  }

  def readFile(path: String, hc: HailContext, buffSize: Int): HadoopPyReader = hc.hadoopConf.readFile(path) { in =>
    new HadoopPyReader(hc.hadoopConf.unsafeReader(path), buffSize)
  }

  def writeFile(path: String, hc: HailContext, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && hc.hadoopConf.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(hc.hadoopConf.unsafeWriter(path))
  }

  def copyFile(from: String, to: String, hc: HailContext) {
    hc.hadoopConf.copy(from, to)
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

  def joinGlobals(left: Table, right: Table, identifier: String): Table = {
    left.annotateGlobal(right.globals.value, right.globalSignature, identifier)
  }

  def joinGlobals(left: Table, right: MatrixTable, identifier: String): Table = {
    left.annotateGlobal(right.globals.value, right.globalType, identifier)
  }

  def joinGlobals(left: MatrixTable, right: Table, identifier: String): MatrixTable = {
    left.annotateGlobal(right.globals.value, right.globalSignature, identifier)
  }

  def joinGlobals(left: MatrixTable, right: MatrixTable, identifier: String): MatrixTable = {
    left.annotateGlobal(right.globals.value, right.globalType, identifier)
  }

  def escapePyString(s: String): String = StringEscapeUtils.escapeString(s)

  def escapeIdentifier(s: String): String = prettyIdentifier(s)

  def fileExists(hc: HailContext, path: String): Boolean = hc.hadoopConf.exists(path) && hc.hadoopConf.isFile(path)

  def dirExists(hc: HailContext, path: String): Boolean = hc.hadoopConf.exists(path) && hc.hadoopConf.isDir(path)

  def mkdir(hc: HailContext, path: String): Boolean = hc.hadoopConf.mkDir(path)

  def copyToTmp(hc: HailContext, path: String, extension: String): String = {
    val codecExt = hc.hadoopConf.getCodec(path)
    val tmpFile = hc.getTemporaryFile(suffix = Some(extension + codecExt))
    hc.hadoopConf.copy(path, tmpFile)
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
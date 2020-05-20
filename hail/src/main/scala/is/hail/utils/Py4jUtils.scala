package is.hail.utils

import java.io.{InputStream, OutputStream}

import is.hail.HailContext
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex}
import is.hail.expr.ir.{ExecuteContext, TableIR, TableLiteral, TableValue}
import is.hail.types.physical.PStruct
import is.hail.types.virtual.{TArray, TString, TStruct, Type}
import is.hail.io.fs.{FS, FileStatus}
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import org.apache.spark.sql.{DataFrame, Row}
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

  def ls(fs: FS, path: String): String = {
    val statuses = fs.listStatus(path)
    JsonMethods.compact(JArray(statuses.map(fs => statusToJson(fs)).toList))
  }

  def stat(fs: FS, path: String): String = {
    val stat = fs.fileStatus(path)
    JsonMethods.compact(statusToJson(stat))
  }

  private def statusToJson(fs: FileStatus): JObject = {
    JObject(
      "path" -> JString(fs.getPath.toString),
      "size_bytes" -> JInt(fs.getLen),
      "size" -> JString(readableBytes(fs.getLen)),
      "is_dir" -> JBool(fs.isDirectory),
      "modification_time" ->
        (if (fs.getModificationTime != null)
          JString(new java.util.Date(fs.getModificationTime).toString)
        else
          JNull),
      "owner" -> (
        if (fs.getOwner != null)
          JString(fs.getOwner)
        else
          JNull))
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

  def readFile(fs: FS, path: String, buffSize: Int): HadoopPyReader =
    new HadoopPyReader(fs.open(path), buffSize)

  def writeFile(fs: FS, path: String, exclusive: Boolean): HadoopPyWriter = {
    if (exclusive && fs.exists(path))
      fatal(s"a file already exists at '$path'")
    new HadoopPyWriter(fs.create(path))
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

  def makeJSON(t: Type, value: Any): String = {
    val jv = JSONAnnotationImpex.exportAnnotation(value, t)
    JsonMethods.compact(jv)
  }

  def importFamJSON(fs: FS, path: String, isQuantPheno: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): String = {
    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)
    val (data, ptyp) = LoadPlink.parseFam(fs, path, ffConfig)
    val jv = JSONAnnotationImpex.exportAnnotation(
      Row(ptyp.virtualType.toString, data),
      TStruct("type" -> TString, "data" -> TArray(ptyp.virtualType)))
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

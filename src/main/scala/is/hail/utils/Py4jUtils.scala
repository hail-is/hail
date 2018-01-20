package is.hail.utils

import java.io.{InputStream, OutputStream}
import java.net.URI
import java.util.IllegalFormatConversionException

import breeze.linalg.{DenseMatrix => BDM, _}
import is.hail.HailContext
import is.hail.annotations.{Memory, Region, RegionValueBuilder}
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.variant.{ReferenceGenome$, Locus, MatrixTable}
import org.apache.commons.io.IOUtils

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

  def bdmGetBytes(bdm: BDM[Double], start: Int, n: Int): Array[Byte] = {
    assert(8L * n < Integer.MAX_VALUE)
    assert(bdm.offset == 0)
    assert(bdm.majorStride == (if (bdm.isTranspose) bdm.cols else bdm.rows))
    val buf = new Array[Byte](8 * n)
    Memory.memcpy(buf, 0, bdm.data, start, n)
    buf
  }

  def getURI(uri: String): String = new URI(uri).getPath

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

  def readFile(path: String, hc: HailContext, buffSize: Int): HadoopPyReader = hc.hadoopConf.readFile(path) { in =>
    new HadoopPyReader(hc.hadoopConf.unsafeReader(path), buffSize)
  }

  def writeFile(path: String, hc: HailContext): HadoopPyWriter = {
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
    left.annotateGlobal(right.globals, right.globalSignature, identifier)
  }

  def joinGlobals(left: Table, right: MatrixTable, identifier: String): Table = {
    left.annotateGlobal(right.globals, right.globalType, identifier)
  }

  def joinGlobals(left: MatrixTable, right: Table, identifier: String): MatrixTable = {
    left.annotateGlobal(right.globals, right.globalSignature, "global." + identifier)
  }

  def joinGlobals(left: MatrixTable, right: MatrixTable, identifier: String): MatrixTable = {
    left.annotateGlobal(right.globals, right.globalType, "global." + identifier)
  }

  def escapePyString(s: String): String = StringEscapeUtils.escapeString(s)

  def makePrintConfig(config: java.util.HashMap[String, String]): PrintConfig = {
    assert(config.size() == 4)
    val pc = PrintConfig(missing = config.get("missing"),
      boolTrue = config.get("bool_true"),
      boolFalse = config.get("bool_false"),
      floatFormat = config.get("float_format"))

    try {
      val a = Array(0.0, 1.0, -1.0)
      val x = a.map(_.formatted(pc.floatFormat))
      if (a.zip(x).exists { case (f, str) => D_!=(f, str.toDouble) }) {
        throw new HailException(
          s"""'float_format' parameter '${ pc.floatFormat }' seems to be invalid.
             |    If this should be a valid format string, please post an issue:
             |      https://github.com/hail-is/hail/issues
             |    Examples of acceptable format strings applied to float 123.456789:
             |      '%.2f' - 123.45
             |      '%.2e' - 1.23e+02""".stripMargin)
      }
    } catch {
      case e: IllegalFormatConversionException =>
        throw new HailException(s"Format string '${ pc.floatFormat }' was invalid", cause = e)
    }

    pc
  }

  def escapeIdentifier(s: String): String = prettyIdentifier(s)
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
package is.hail.utils

import java.io.{InputStream, OutputStream}
import java.net.URI

import breeze.linalg.{DenseMatrix => BDM, _}
import is.hail.HailContext
import is.hail.annotations.{Memory, Region, RegionValueBuilder}
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.variant.{GenomeReference, Locus, MatrixTable}

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

  def readFile(path: String, hc: HailContext): HadoopPyReader = hc.hadoopConf.readFile(path) { in =>
    new HadoopPyReader(hc.hadoopConf.unsafeReader(path))
  }

  def readBinaryFile(path: String, hc: HailContext): HadoopPyBinaryReader = hc.hadoopConf.readFile(path) { in =>
    new HadoopPyBinaryReader(hc.hadoopConf.unsafeReader(path))
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
    left.annotateGlobal(right.globalAnnotation, right.globalSignature, identifier)
  }

  def joinGlobals(left: MatrixTable, right: Table, identifier: String): MatrixTable = {
    left.annotateGlobal(right.globals, right.globalSignature, "global." + identifier)
  }

  def joinGlobals(left: MatrixTable, right: MatrixTable, identifier: String): MatrixTable = {
    left.annotateGlobal(right.globalAnnotation, right.globalSignature, "global." + identifier)
  }

  def escapePyString(s: String): String = StringEscapeUtils.escapeString(s)
}

class HadoopPyReader(in: InputStream) {
  var eof = false

  def read(n: Int): String = {
    val b = new Array[Byte](n)
    val bytesRead = in.read(b, 0, n)
    if (bytesRead < 0)
      new String(new Array[Byte](0), "ISO-8859-1")
    else if (bytesRead == n)
      new String(b, "ISO-8859-1")
    else
      new String(b.slice(0, bytesRead), "ISO-8859-1")
  }

  def close() {
    in.close()
  }
}

class HadoopPyBinaryReader(in: InputStream) {
  var eof = false

  def read(n: Int): Array[Byte] = {
    val b = new Array[Byte](n)
    val bytesRead = in.read(b, 0, n)
    if (bytesRead < 0)
      new Array[Byte](0)
    else if (bytesRead == n)
      b
    else
      b.slice(0, bytesRead)
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
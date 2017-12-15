package is.hail.utils

import java.io.{InputStream, OutputStream}

import is.hail.HailContext
import is.hail.keytable.Table
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

  def parseIntervalList(strs: java.util.ArrayList[String], gr: GenomeReference): IntervalTree[Locus, Unit] = {
    implicit val locusOrd = gr.locusOrdering
    IntervalTree(Locus.parseIntervals(strs.asScala.toArray, gr))
  }


  def makeIntervalList(intervals: java.util.ArrayList[Interval[Locus]], gr: GenomeReference): IntervalTree[Locus, Unit] = {
    implicit val locusOrd = gr.locusOrdering
    IntervalTree(intervals.asScala.toArray)
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

  def readFile(path: String, hc: HailContext): HadoopPyReader = hc.hadoopConf.readFile(path) { in =>
    new HadoopPyReader(hc.hadoopConf.unsafeReader(path))
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
package is.hail.utils

import java.io.{InputStream, OutputStreamWriter}
import java.util

import is.hail.HailContext
import is.hail.utils._
import is.hail.variant.Locus

import scala.collection.JavaConverters._
import scala.io.{BufferedSource, Source}

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

  def parseIntervalList(strs: java.util.ArrayList[String]): IntervalTree[Locus] =
    IntervalTree(Locus.parseIntervals(strs.asScala.toArray), prune = true)

  def makeIntervalList(intervals: java.util.ArrayList[Interval[Locus]]): IntervalTree[Locus] =
    IntervalTree(intervals.asScala.toArray, prune = true)

  // FIXME: don't use vector
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
}

class HadoopPyReader(in: InputStream) {
  var eof = false

  def read(n: Int): String = {
    if (eof)
      new String(new Array[Byte](0), "ISO-8859-1")
    else {
      var i = 0
      val b = new Array[Byte](n)
      while (!eof && i < n) {
        val bytesRead = in.read(b, i, b.length - i)
        if (bytesRead < 0) {
          eof = true
        } else
          i += bytesRead
      }

      if (eof)
        new String(b.slice(0, i), "ISO-8859-1")
      else
        new String(b, "ISO-8859-1")
    }
  }

  def close() {
    in.close()
  }
}

class HadoopPyWriter(out: OutputStreamWriter) {

  def write(s: String) {
    out.write(s)
  }

  def close() {
    out.close()
  }
}
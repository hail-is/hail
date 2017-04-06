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

  def readFile(path: String, hc: HailContext, buffer: Int): HadoopPyReader = hc.hadoopConf.readFile(path) { in =>
    new HadoopPyReader(hc.hadoopConf.unsafeReader(path), buffer)
  }

  def writeFile(path: String, hc: HailContext): HadoopPyWriter = {
    new HadoopPyWriter(hc.hadoopConf.unsafeWriter(path))
  }

  def copyFile(from: String, to: String, hc: HailContext) {
    hc.hadoopConf.copy(from, to)
  }
}

class HadoopPyReader(in: InputStream, buffer: Int) {
  private val lines = Source.fromInputStream(in).getLines()

  def close() {
    in.close()
  }

  def readFully(): String = {
    lines.mkString("\n")
  }

  def readChunk(): String = {
    if (lines.isEmpty)
      null
    else
      lines.take(buffer).mkString("\n")
  }
}

class HadoopPyWriter(out: OutputStreamWriter) {

  def write(s: String) {
    out.write(s)
  }

  def writeLine(s: String) {
    out.write(s)
    out.write('\n')
  }

  def close() {
    out.close()
  }
}
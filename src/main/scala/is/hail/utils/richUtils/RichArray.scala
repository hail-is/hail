package is.hail.utils.richUtils

import is.hail.HailContext
import is.hail.io.{DoubleInputBuffer, DoubleOutputBuffer}
import is.hail.utils._

object RichArray {
  val defaultBufSize: Int = 4096 << 3
  
  def importFromDoubles(hc: HailContext, path: String, n: Int): Array[Double] = {
    val a = new Array[Double](n)
    importFromDoubles(hc, path, a, defaultBufSize)
    a
  }
  
  def importFromDoubles(hc: HailContext, path: String, a: Array[Double], bufSize: Int): Unit = {
    hc.hadoopConf.readFile(path) { is =>
      val in = new DoubleInputBuffer(is, bufSize)

      in.readDoubles(a)
    }
  }

  def exportToDoubles(hc: HailContext, path: String, a: Array[Double]): Unit =
    exportToDoubles(hc, path, a, defaultBufSize)

  def exportToDoubles(hc: HailContext, path: String, a: Array[Double], bufSize: Int): Unit = {
    hc.hadoopConf.writeFile(path) { os =>
      val out = new DoubleOutputBuffer(os, bufSize)

      out.writeDoubles(a)
      out.flush()
    }
  }
}

class RichArray[T](val a: Array[T]) extends AnyVal {
  def index: Map[T, Int] = a.zipWithIndex.toMap
}

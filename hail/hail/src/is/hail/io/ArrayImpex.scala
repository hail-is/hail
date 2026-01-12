package is.hail.io

import is.hail.io.fs.FS
import is.hail.utils.using

object ArrayImpex {
  val DefaultBuffSize: Int = 4096 << 3

  def importFromDoubles(fs: FS, path: String, n: Int): Array[Double] = {
    val a = new Array[Double](n)
    importFromDoubles(fs, path, a, DefaultBuffSize)
    a
  }

  def importFromDoubles(fs: FS, path: String, a: Array[Double], bufSize: Int): Unit =
    using(new DoubleInputBuffer(fs.open(path), bufSize))(_.readDoubles(a))

  def exportToDoubles(fs: FS, path: String, a: Array[Double]): Unit =
    exportToDoubles(fs, path, a, DefaultBuffSize)

  def exportToDoubles(fs: FS, path: String, a: Array[Double], bufSize: Int): Unit =
    using(new DoubleOutputBuffer(fs.create(path), bufSize))(_.writeDoubles(a))
}

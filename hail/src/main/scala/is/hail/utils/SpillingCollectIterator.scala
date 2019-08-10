package is.hail.utils

import is.hail.HailContext
import java.io.{ ObjectInputStream, ObjectOutputStream }
import org.apache.spark.rdd.RDD
import scala.reflect.ClassTag
import scala.reflect.classTag

object SpillingCollectIterator {
  def apply[T: ClassTag](rdd: RDD[T], sizeLimit: Int): SpillingCollectIterator[T] = {
    val x = new SpillingCollectIterator(rdd, sizeLimit)
    val ctc = classTag[T]
    HailContext.get.sc.runJob(
      rdd,
      (_, it: Iterator[T]) => it.toArray(ctc),
      0 until rdd.partitions.length,
      x.append _)
    x
  }
}

class SpillingCollectIterator[T: ClassTag] private (rdd: RDD[T], sizeLimit: Int) extends Iterator[T] {
  private[this] val hc = HailContext.get
  private[this] val fs = hc.sFS
  private[this] val sc = hc.sc
  private[this] val files: Array[(String, Long)] = new Array(rdd.partitions.length)
  private[this] var buf: Array[Array[T]] = new Array(rdd.partitions.length)
  private[this] var size: Long = 0L
  private[this] var i: Int = 0
  private[this] var it: Iterator[T] = null

  private def append(partition: Int, a: Array[T]): Unit = synchronized {
    assert(buf(partition) == null)
    buf(partition) = a
    size += a.length
    if (size > sizeLimit) {
      val file = hc.getTemporaryFile()
      fs.writeFileNoCompression(file) { os =>
        var k = 0
        while (k < buf.length) {
          val vals = buf(k)
          if (vals != null) {
            buf(k) = null
            val pos = os.getPos
            val oos = new ObjectOutputStream(os)
            oos.writeInt(vals.length)
            var j = 0
            while (j < vals.length) {
              oos.writeObject(vals(j))
              j += 1
            }
            files(k) = (file, pos)
            oos.flush()
          }
          k += 1
        }
      }
      size = 0
    }
  }

  def hasNext: Boolean = {
    if (it == null || !it.hasNext) {
      if (i >= files.length) {
        it = null
        return false
      } else if (files(i) == null) {
        assert(buf(i) != null)
        it = buf(i).iterator
        buf(i) = null
      } else {
        val (filename, pos) = files(i)
        fs.readFileNoCompression(filename) { is =>
          is.seek(pos)
          using(new ObjectInputStream(is)) { ois =>
            val length = ois.readInt()
            val arr = new Array[T](length)
            var j = 0
            while (j < length) {
              arr(j) = ois.readObject().asInstanceOf[T]
              j += 1
            }
            it = arr.iterator
          }
        }
      }
      i += 1
    }
    it.hasNext
  }

  def next: T = {
    hasNext
    it.next
  }
}


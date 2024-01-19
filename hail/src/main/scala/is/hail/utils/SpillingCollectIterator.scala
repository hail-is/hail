package is.hail.utils

import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkBackend
import is.hail.io.fs.FS

import scala.reflect.{classTag, ClassTag}

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.spark.rdd.RDD

object SpillingCollectIterator {
  def apply[T: ClassTag](localTmpdir: String, fs: FS, rdd: RDD[T], sizeLimit: Int)
    : SpillingCollectIterator[T] = {
    val nPartitions = rdd.partitions.length
    val x = new SpillingCollectIterator(localTmpdir, fs, nPartitions, sizeLimit)
    val ctc = classTag[T]
    SparkBackend.sparkContext("SpillingCollectIterator.apply").runJob(
      rdd,
      (_, it: Iterator[T]) => it.toArray(ctc),
      0 until nPartitions,
      x.append _,
    )
    x
  }
}

class SpillingCollectIterator[T: ClassTag] private (
  localTmpdir: String,
  fs: FS,
  nPartitions: Int,
  sizeLimit: Int,
) extends Iterator[T] {
  private[this] val files: Array[(String, Long)] = new Array(nPartitions)
  private[this] val buf: Array[Array[T]] = new Array(nPartitions)
  private[this] var _size: Long = 0L
  private[this] var i: Int = 0
  private[this] var it: Iterator[T] = null

  private def append(partition: Int, a: Array[T]): Unit = synchronized {
    assert(buf(partition) == null)
    buf(partition) = a
    _size += a.length
    if (_size > sizeLimit) {
      val file =
        ExecuteContext.createTmpPathNoCleanup(localTmpdir, s"spilling-collect-iterator-$partition")
      log.info(s"spilling partition $partition to $file")
      using(fs.createNoCompression(file)) { os =>
        var k = 0
        while (k < buf.length) {
          val vals = buf(k)
          if (vals != null) {
            buf(k) = null
            val pos = os.getPosition
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
      _size = 0
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
        using(fs.openNoCompression(filename)) { is =>
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

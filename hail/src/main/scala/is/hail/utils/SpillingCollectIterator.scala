package is.hail.utils

import is.hail.HailContext
import java.io.{ ObjectInputStream, ObjectOutputStream }
import java.util.TreeMap
import java.util.function.BiConsumer
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.classTag

object SpillingCollectIterator {
  def apply[T: ClassTag](rdd: RDD[T], sizeLimit: Int = 1000): SpillingCollectIterator[T] = {
    val x = new SpillingCollectIterator(rdd, sizeLimit)
    x.runJob()
    x
  }
}

class SpillingCollectIterator[T: ClassTag] private (rdd: RDD[T], sizeLimit: Int) extends Iterator[T] {
  private[this] val hc = HailContext.get
  private[this] val hConf = hc.hadoopConf
  private[this] val sc = hc.sc
  private[this] val files: TreeMap[Int, (Int, String, Long)] = new TreeMap()
  private[this] val buf: TreeMap[Int, (Int, Array[T])] = new TreeMap()
  private[this] var size: Long = 0L
  private[this] var i: Int = -1
  private[this] var it: Iterator[T] = null
  private[this] var readyToIterate: Boolean = false

  private def runJob(): Unit = {
    val ctc = classTag[T]
    sc.runJob(
      rdd,
      (_, it: Iterator[T]) => it.toArray(ctc),
      0 until rdd.partitions.length,
      append _)
    readyToIterate = true
  }

  private[this] def append(partition: Int, a: Array[T]): Unit = synchronized {
    assert(buf.get(partition) == null)
    var l = partition
    var r = partition + 1
    val ab = new ArrayBuilder[T]()
    val prev = buf.floorEntry(partition)
    if (prev != null) {
      val prevL = prev.getKey()
      val (prevR, prevVals) = prev.getValue()
      if (prevR == partition) {
        ab ++= prevVals
        l = prevL
        buf.remove(prevL)
      }
    }
    ab ++= a
    val next = buf.ceilingEntry(partition)
    if (next != null) {
      val nextL = next.getKey()
      val (nextR, nextVals) = next.getValue()
      if (nextL == partition + 1) {
        ab ++= nextVals
        r = nextR
        buf.remove(nextL)
      }
    }
    val newVals = ab.result()
    buf.put(l, (r, newVals))
    size += a.length
    if (size > sizeLimit) {
      val file = hc.getTemporaryFile()
      hConf.writeFileNoCompression(file) { os =>
        buf.forEach(new BiConsumer[Int, (Int, Array[T])]() {
          def accept(l: Int, p: (Int, Array[T])): Unit = {
            val pos = os.getPos
            val oos = new ObjectOutputStream(os)
            val (r, vals) = p
            oos.writeInt(l)
            oos.writeInt(r)
            oos.writeInt(vals.length)
            var j = 0
            while (j < vals.length) {
              oos.writeObject(vals(j))
              j += 1
            }
            files.put(l, (r, file, pos))
            oos.flush()
          }
        })
      }
      buf.clear()
      size = 0
    }
  }

  def hasNext: Boolean = {
    assert(readyToIterate)
    if (it == null || !it.hasNext) {
      if (i == -1) {
        i += 1
        if (!buf.isEmpty()) {
          it = buf.values().iterator.map(_._2).flatten
          return it.hasNext
        }
      }
      buf.clear()
      if (i < files.size()) {
        val glb = files.floorEntry(i)
        val l = glb.getKey()
        val (r, filename, pos) = glb.getValue()
        assert(l <= i && i < r, s"$l $i $r")
        hConf.readFileNoCompression(filename) { is =>
          is.seek(pos)
          using(new ObjectInputStream(is)) { ois =>
            ois.readInt()
            ois.readInt()
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
        i += 1
        return it.hasNext
      }
      i += 1
      it = null
      return false
    }
    return it != null && it.hasNext
  }

  def next: T = {
    assert(readyToIterate)
    hasNext
    it.next
  }
}


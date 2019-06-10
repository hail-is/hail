package is.hail.utils

import is.hail.HailContext
import java.io.{ ObjectInputStream, ObjectOutputStream }
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.classTag

object SpillingCollectIterator {
  def apply[T: ClassTag](rdd: RDD[T], sizeLimit: Int = 1000): SpillingCollectIterator[T] = {
    val x = new SpillingCollectIterator(rdd, sizeLimit)
    x.runJob()
    x
  }

  private def iteratorToArray[T](ctc: ClassTag[T])(tctx: TaskContext, it: Iterator[T]): Array[T] =
    it.toArray(ctc)
}

class SpillingCollectIterator[T: ClassTag] private (rdd: RDD[T], sizeLimit: Int) extends Iterator[T] {
  private[this] val hc = HailContext.get
  private[this] val hConf = hc.hadoopConf
  private[this] val sc = hc.sc
  private[this] val files: ArrayBuffer[String] = new ArrayBuffer[String]()
  private[this] val buf: ArrayBuffer[T] = new ArrayBuffer()
  private[this] var i: Int = -1
  private[this] var it: Iterator[T] = null
  private[this] var readyToIterate: Boolean = false

  private def runJob(): Unit = {
    sc.runJob(rdd, SpillingCollectIterator.iteratorToArray(classTag[T]), 0 until rdd.partitions.length,
      (partition, a: Array[T]) => append(a))
    readyToIterate = true
  }

  private[this] def append(a: Array[T]): Unit = synchronized {
    buf ++= a
    if (buf.length == sizeLimit) {
      val file = hc.getTemporaryFile()
      hConf.writeFile(file) { os =>
	      using(new ObjectOutputStream(os)) { oos =>
          var j = 0
          while (j < buf.length) {
            oos.writeObject(buf(j))
            j += 1
          }
        }
      }
	    files += file
      buf.clear()
    }
  }

  def hasNext: Boolean = {
    assert(readyToIterate)
    if (it == null || !it.hasNext) {
      if (i == -1) {
        i += 1
        if (buf.nonEmpty) {
          it = buf.iterator
          return it.hasNext
        }
      }
      if (i < files.length) {
        buf.clear()
        hConf.readFile(files(i)) { is =>
	        using(new ObjectInputStream(is)) { ois =>
	          var j = 0
	          while (j < sizeLimit) {
	            buf += ois.readObject().asInstanceOf[T]
	            j += 1
	          }
	        }
	      }
        i += 1
        it = buf.iterator
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


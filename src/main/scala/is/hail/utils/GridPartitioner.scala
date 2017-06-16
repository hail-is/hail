package is.hail.utils

import org.apache.spark.Partitioner
import scala.reflect.classTag

object GridPartitioner {

  def apply(nRowBlocks: Int, nColBlocks: Int): Partitioner = {
    val intClass = classTag[Int].runtimeClass
    val gpObjectClass = Class.forName("org.apache.spark.mllib.linalg.distributed.GridPartitioner$")
    val gpApply = gpObjectClass.getMethod("apply", intClass, intClass, intClass, intClass)

    try {
      gpApply.setAccessible(true)
      gpApply.invoke(gpObjectClass.getField("MODULE$").get(null), nRowBlocks: java.lang.Integer,
        nColBlocks: java.lang.Integer, 1: java.lang.Integer, 1: java.lang.Integer).asInstanceOf[Partitioner]
    } finally {
      gpApply.setAccessible(false)
    }
  }
}

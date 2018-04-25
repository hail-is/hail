package is.hail.utils.richUtils

import java.io._

import org.apache.spark.TaskContext
import is.hail.utils._
import is.hail.sparkextras._

import scala.reflect.ClassTag

class RichContextRDD[C <: AutoCloseable, T: ClassTag](crdd: ContextRDD[C, T]) {
  def writePartitions(path: String,
    write: (Int, Iterator[T], OutputStream) => Long): (Array[String], Array[Long]) = {
    val sc = crdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/parts")

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val (partFiles, partitionCounts) = crdd.mapPartitionsWithIndex { case (i, it) =>
      val f = partFile(d, i, TaskContext.get)
      val filename = path + "/parts/" + f
      val os = sHadoopConfBc.value.value.unsafeWriter(filename)
      Iterator.single(f -> write(i, it, os))
    }
      .collect()
      .unzip

    val itemCount = partitionCounts.sum
    assert(nPartitions == partitionCounts.length)

    info(s"wrote $itemCount ${ plural(itemCount, "item") } " +
      s"in ${ nPartitions } ${ plural(nPartitions, "partition") }")

    (partFiles, partitionCounts)
  }
}

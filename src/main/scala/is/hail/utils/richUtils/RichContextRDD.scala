package is.hail.utils.richUtils

import java.io._

import org.apache.commons.lang3.StringUtils
import org.apache.spark.TaskContext
import is.hail.utils._
import is.hail.sparkextras._

import scala.reflect.ClassTag

class RichContextRDD[C <: ResettableContext, T: ClassTag](crdd: ContextRDD[C, T]) {
  def writePartitions(path: String,
    write: (Int, Iterator[T], OutputStream) => Long,
    remapPartitions: Option[(Array[Int], Int)] = None): (Array[String], Array[Long]) = {
    val sc = crdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/parts")

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitionsToWrite = crdd.getNumPartitions

    val (remap, nPartitions) = remapPartitions match {
      case Some((map, n)) => (map.apply _, n)
      case None => (identity[Int] _, nPartitionsToWrite)
    }

    val d = digitsNeeded(nPartitions)

    val remapBc = sc.broadcast(remap)

    val (partFiles, partitionCounts) = crdd.mapPartitionsWithIndex { case (index, it) =>
      val i = remapBc.value(index)
      val f = partFile(d, i, TaskContext.get)
      val filename = path + "/parts/" + f
      val os = sHadoopConfBc.value.value.unsafeWriter(filename)
      Iterator.single(f -> write(i, it, os))
    }
      .collect()
      .unzip

    val itemCount = partitionCounts.sum
    assert(nPartitionsToWrite == partitionCounts.length)

    info(s"wrote $itemCount ${ plural(itemCount, "item") } " +
      s"in ${ nPartitionsToWrite } ${ plural(nPartitionsToWrite, "partition") }")

    (partFiles, partitionCounts)
  }
}

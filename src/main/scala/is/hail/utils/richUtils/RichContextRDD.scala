package is.hail.utils.richUtils

import java.io._

import org.apache.commons.lang3.StringUtils
import is.hail.utils._
import is.hail.sparkextras._

import scala.reflect.ClassTag

class RichContextRDD[C <: AutoCloseable, T: ClassTag](crdd: ContextRDD[C, T]) {
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

    val partFiles = Array.tabulate[String](nPartitions) { i =>
      val is = i.toString
      assert(is.length <= d)
      "part-" + StringUtils.leftPad(is, d, "0")
    }

    val partitionCounts = crdd.mapPartitionsWithIndex { case (index, it) =>
      val i = remapBc.value(index)
      val filename = path + "/parts/" + partFile(d, i)
      val os = sHadoopConfBc.value.value.unsafeWriter(filename)
      Iterator.single(write(i, it, os))
    }.run.collect()

    val itemCount = partitionCounts.sum
    assert(nPartitionsToWrite == partitionCounts.length)

    info(s"wrote $itemCount ${ plural(itemCount, "item") } " +
      s"in ${ nPartitionsToWrite } ${ plural(nPartitionsToWrite, "partition") }")

    (partFiles, partitionCounts)
  }
}

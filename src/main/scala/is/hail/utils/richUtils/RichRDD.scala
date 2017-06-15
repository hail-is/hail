package is.hail.utils.richUtils

import is.hail.sparkextras.ReorderedPartitionsRDD
import is.hail.utils._
import org.apache.hadoop
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.{NarrowDependency, Partition, TaskContext}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag
import scala.collection.mutable

case class SubsetRDDPartition(index: Int, parentPartition: Partition) extends Partition


class RichRDD[T](val r: RDD[T]) extends AnyVal {
  def countByValueRDD()(implicit tct: ClassTag[T]): RDD[(T, Int)] = r.map((_, 1)).reduceByKey(_ + _)

  def reorderPartitions(oldIndices: Array[Int])(implicit tct: ClassTag[T]): RDD[T] =
    new ReorderedPartitionsRDD[T](r, oldIndices)

  def forall(p: T => Boolean)(implicit tct: ClassTag[T]): Boolean = !exists(x => !p(x))

  def exists(p: T => Boolean)(implicit tct: ClassTag[T]): Boolean = r.mapPartitions { it =>
    Iterator(it.exists(p))
  }.fold(false)(_ || _)

  def writeTable(filename: String, tmpDir: String, header: Option[String] = None, parallelWrite: Boolean = false) {
    val hConf = r.sparkContext.hadoopConfiguration

    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(filename)))

    hConf.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath =
      if (parallelWrite) {
        filename
      } else
        hConf.getTemporaryFile(tmpDir)

    val rWithHeader = header.map { h =>
      if (r.partitions.length == 0)
        r.sparkContext.parallelize(List(h), 1)
      else if (parallelWrite)
        r.mapPartitions { it => Iterator(h) ++ it }
      else
        r.mapPartitionsWithIndex { case (i, it) =>
          if (i == 0)
            Iterator(h) ++ it
          else
            it
        }
    }.getOrElse(r)

    codec match {
      case Some(x) => rWithHeader.saveAsTextFile(parallelOutputPath, x.getClass)
      case None => rWithHeader.saveAsTextFile(parallelOutputPath)
    }

    if (!hConf.exists(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    if (!parallelWrite) {
      hConf.copyMerge(parallelOutputPath, filename, true, false)
    }
  }

  def collectOrdered()(implicit tct: ClassTag[T]): Array[T] =
    r.zipWithIndex().collect().sortBy(_._2).map(_._1)

  def find(f: T => Boolean): Option[T] = r.filter(f).take(1) match {
    case Array(elem) => Some(elem)
    case _ => None
  }

  def collectAsSet(): collection.Set[T] = {
    r.aggregate(mutable.Set.empty[T])(
      { case (s, elem) => s += elem },
      { case (s1, s2) => s1 ++ s2 }
    )
  }

  def subsetPartitions(keep: Array[Int])(implicit ct: ClassTag[T]): RDD[T] = {
    require(keep.length <= r.partitions.length, "tried to subset to more partitions than exist")
    require(keep.isSorted && keep.forall{i => i >= 0 && i < r.partitions.length},
      "values not sorted or not in range [0, number of partitions)")
    val parentPartitions = r.partitions

    new RDD[T](r.sparkContext, Seq(new NarrowDependency[T](r) {
      def getParents(partitionId: Int): Seq[Int] = Seq(keep(partitionId))
    })) {
      def getPartitions: Array[Partition] = keep.indices.map { i =>
        SubsetRDDPartition(i, parentPartitions(keep(i)))
      }.toArray

      def compute(split: Partition, context: TaskContext): Iterator[T] =
        r.compute(split.asInstanceOf[SubsetRDDPartition].parentPartition, context)
    }
  }
}

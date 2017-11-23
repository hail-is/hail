package is.hail.utils.richUtils

import java.io.OutputStream

import is.hail.sparkextras.ReorderedPartitionsRDD
import is.hail.utils._
import org.apache.commons.lang3.StringUtils
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
      if (r.getNumPartitions == 0)
        r.sparkContext.parallelize(List(h), numSlices = 1)
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
      hConf.copyMerge(parallelOutputPath, filename, rWithHeader.getNumPartitions, hasHeader = false)
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
    require(keep.isSorted && keep.forall { i => i >= 0 && i < r.partitions.length },
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

  def countPerPartition()(implicit ct: ClassTag[T]): Array[Long] = {
    val sc = r.sparkContext
    sc.runJob(r, getIteratorSize _)
  }

  def headPerPartition(n: Int)(implicit ct: ClassTag[T]): RDD[T] = {
    require(n >= 0)
    r.mapPartitions(_.take(n), preservesPartitioning = true)
  }

  /**
    * Parts of this method are lifted from:
    *   org.apache.spark.rdd.RDD.take
    *   Spark version 2.0.2
    */
  def head(n: Long)(implicit ct: ClassTag[T]): RDD[T] = {
    require(n >= 0)

    val sc = r.sparkContext
    val nPartitions = r.getNumPartitions

    var partScanned = 0
    var nLeft = n
    var idxLast = 0
    var nLast = 0L
    var numPartsToTry = 1L

    while (nLeft > 0 && partScanned < nPartitions) {
      val nSeen = n - nLeft

      if (partScanned > 0) {
        // If we didn't find any rows after the previous iteration, quadruple and retry.
        // Otherwise, interpolate the number of partitions we need to try, but overestimate
        // it by 50%. We also cap the estimation in the end.
        if (nSeen == 0) {
          numPartsToTry = partScanned * 4
        } else {
          // the left side of max is >=1 whenever partsScanned >= 2
          numPartsToTry = Math.max((1.5 * n * partScanned / nSeen).toInt - partScanned, 1)
          numPartsToTry = Math.min(numPartsToTry, partScanned * 4)
        }
      }

      val p = partScanned.until(math.min(partScanned + numPartsToTry, nPartitions).toInt)
      val counts = sc.runJob(r, getIteratorSizeWithMaxN(nLeft) _, p)

      p.zip(counts).foreach { case (idx, c) =>
        if (nLeft > 0) {
          idxLast = idx
          nLast = if (c < nLeft) c else nLeft
          nLeft -= nLast
        }
      }

      partScanned += p.size
    }

    r.mapPartitionsWithIndex({ case (i, it) =>
      if (i == idxLast)
        it.take(nLast.toInt)
      else
        it
    }, preservesPartitioning = true)
      .subsetPartitions((0 to idxLast).toArray)
  }
  
  def writePartitions(path: String, write: (Int, Iterator[T], OutputStream) => Long): Long = {
    
    val sc = r.sparkContext
    val hadoopConf = sc.hadoopConfiguration
    
    hadoopConf.mkDir(path + "/parts")
    
    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))
    
    val nPartitions = r.getNumPartitions
    val d = digitsNeeded(nPartitions)

    val itemCount = r.mapPartitionsWithIndex { case (i, it) =>
      val is = i.toString
      assert(is.length <= d)
      val pis = StringUtils.leftPad(is, d, "0")

      val filename = path + "/parts/part-" + pis
      
      val os = sHadoopConfBc.value.value.unsafeWriter(filename)

      Iterator.single(write(i, it, os))
    }
      .fold(0L)(_ + _)

    info(s"wrote $itemCount items in $nPartitions partitions")
    
    itemCount
  }
  
  // FIXME: persist issues?
  // returns cumulative index of first element in each partition and total number of elements
  // example: three partitions with |P1| = 3, |P2| = 5, |P3| = 4; returns [0, 3, 8, 12]
  def computePartitionBoundaries(): Array[Long] =
    r.mapPartitions(it => Iterator(it.length), preservesPartitioning = true)
      .collect()
      .scanLeft(0L)(_ + _)
}

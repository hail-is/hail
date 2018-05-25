package is.hail.utils.richUtils

import java.io.OutputStream

import is.hail.rvd.RVDContext
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.hadoop
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.{NarrowDependency, Partition, Partitioner, TaskContext}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag
import scala.collection.mutable

case class SubsetRDDPartition(index: Int, parentPartition: Partition) extends Partition

case class SupersetRDDPartition(index: Int, maybeParentPartition: Option[Partition]) extends Partition

class RichRDD[T](val r: RDD[T]) extends AnyVal {
  def reorderPartitions(oldIndices: Array[Int])(implicit tct: ClassTag[T]): RDD[T] =
    new ReorderedPartitionsRDD[T](r, oldIndices)

  def forall(p: T => Boolean)(implicit tct: ClassTag[T]): Boolean = !exists(x => !p(x))

  def exists(p: T => Boolean)(implicit tct: ClassTag[T]): Boolean = r.mapPartitions { it =>
    Iterator(it.exists(p))
  }.fold(false)(_ || _)

  def writeTable(filename: String, tmpDir: String, header: Option[String] = None, exportType: Int = ExportType.CONCATENATED) {
    val hConf = r.sparkContext.hadoopConfiguration

    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(filename)))

    hConf.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath =
      if (exportType == ExportType.CONCATENATED)
        hConf.getTemporaryFile(tmpDir)
      else
        filename

    val rWithHeader: RDD[_] = header.map { h =>
      if (r.getNumPartitions == 0 && exportType != ExportType.PARALLEL_SEPARATE_HEADER)
        r.sparkContext.parallelize(List(h), numSlices = 1)
      else {
        exportType match {
          case ExportType.CONCATENATED =>
            r.mapPartitionsWithIndex { case (i, it) =>
              if (i == 0)
                Iterator(h) ++ it
              else
                it
            }
          case ExportType.PARALLEL_SEPARATE_HEADER =>
            r
          case ExportType.PARALLEL_HEADER_IN_SHARD =>
            r.mapPartitions { it => Iterator(h) ++ it }
          case _ => fatal(s"Unknown export type: $exportType")
        }
      }
    }.getOrElse(r)

    codec match {
      case Some(x) => rWithHeader.saveAsTextFile(parallelOutputPath, x.getClass)
      case None => rWithHeader.saveAsTextFile(parallelOutputPath)
    }

    if (exportType == ExportType.PARALLEL_SEPARATE_HEADER) {
      val headerExt = hConf.getCodec(filename)
      hConf.writeTextFile(parallelOutputPath + "/header" + headerExt) { out =>
        header.foreach { h =>
          out.write(h)
          out.write('\n')
        }
      }
    }

    if (!hConf.exists(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    if (exportType == ExportType.CONCATENATED) {
      hConf.copyMerge(parallelOutputPath, filename, rWithHeader.getNumPartitions, header = false)
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

  def subsetPartitions(keep: Array[Int], newPartitioner: Option[Partitioner] = None)(implicit ct: ClassTag[T]): RDD[T] = {
    require(keep.length <= r.partitions.length,
      s"tried to subset to more partitions than exist ${keep.toSeq} ${r.partitions.toSeq}")
    require(keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < r.partitions.length)),
      "values not sorted or not in range [0, number of partitions)")
    val parentPartitions = r.partitions

    new RDD[T](r.sparkContext, FastSeq(new NarrowDependency[T](r) {
      def getParents(partitionId: Int): Seq[Int] = FastSeq(keep(partitionId))
    })) {
      def getPartitions: Array[Partition] = keep.indices.map { i =>
        SubsetRDDPartition(i, parentPartitions(keep(i)))
      }.toArray

      def compute(split: Partition, context: TaskContext): Iterator[T] =
        r.compute(split.asInstanceOf[SubsetRDDPartition].parentPartition, context)
      
      @transient override val partitioner: Option[Partitioner] = newPartitioner
    }
  }

  def supersetPartitions(
    oldToNewPI: Array[Int],
    newNPartitions: Int,
    newPIPartition: Int => Iterator[T],
    newPartitioner: Option[Partitioner] = None)(implicit ct: ClassTag[T]): RDD[T] = {
    
    require(oldToNewPI.length == r.partitions.length)
    require(oldToNewPI.forall(pi => pi >= 0 && pi < newNPartitions))
    require(oldToNewPI.areDistinct())
    
    val parentPartitions = r.partitions
    val newToOldPI = oldToNewPI.zipWithIndex.toMap

    new RDD[T](r.sparkContext, FastSeq(new NarrowDependency[T](r) {
      def getParents(partitionId: Int): Seq[Int] = newToOldPI.get(partitionId) match {
        case Some(oldPI) => Array(oldPI)
        case None => Array.empty[Int]
      }
    })) {
      def getPartitions: Array[Partition] = Array.tabulate(newNPartitions) { i =>
        SupersetRDDPartition(i, newToOldPI.get(i).map(parentPartitions))
      }

      def compute(split: Partition, context: TaskContext): Iterator[T] = {
        split.asInstanceOf[SupersetRDDPartition].maybeParentPartition match {
          case Some(part) => r.compute(part, context)
          case None => newPIPartition(split.index)
        }
      }

      @transient override val partitioner: Option[Partitioner] = newPartitioner
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
    var idxLast = -1
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

  def writePartitions(path: String,
    write: (Iterator[T], OutputStream) => Long
  )(implicit tct: ClassTag[T]
  ): (Array[String], Array[Long]) =
    ContextRDD.weaken[RVDContext](r).writePartitions(
      path,
      (_, it, os) => write(it, os))
}

package is.hail.utils.richUtils

import is.hail.backend.ExecuteContext

import java.io.{OutputStream, OutputStreamWriter}
import is.hail.io.FileWriteMetadata
import is.hail.rvd.RVDContext
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.io.compress.{BGzipCodec, ComposableBGzipCodec, ComposableBGzipOutputStream}
import is.hail.io.fs.FS
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

  def writeTable(ctx: ExecuteContext, filename: String, header: Option[String] = None, exportType: String = ExportType.CONCATENATED) {
    val hConf = r.sparkContext.hadoopConfiguration
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = {
      val codec = codecFactory.getCodec(new hadoop.fs.Path(filename))
      if (codec != null && codec.isInstanceOf[BGzipCodec] && exportType == ExportType.PARALLEL_COMPOSABLE)
        new ComposableBGzipCodec
      else
        codec
    }

    val fs = ctx.fs
    fs.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath =
      if (exportType == ExportType.CONCATENATED)
        ctx.createTmpPath("write-table-concatenated")
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
          case ExportType.PARALLEL_COMPOSABLE =>
            r
          case ExportType.PARALLEL_HEADER_IN_SHARD =>
            r.mapPartitions { it => Iterator(h) ++ it }
          case _ => fatal(s"Unknown export type: $exportType")
        }
      }
    }.getOrElse(r)

    Option(codec) match {
      case Some(x) => rWithHeader.saveAsTextFile(parallelOutputPath, x.getClass)
      case None => rWithHeader.saveAsTextFile(parallelOutputPath)
    }

    if (exportType == ExportType.PARALLEL_SEPARATE_HEADER) {
      val headerExt = fs.getCodecExtension(filename)
      using(new OutputStreamWriter(fs.create(parallelOutputPath + "/header" + headerExt))) { out =>
        header.foreach { h =>
          out.write(h)
          out.write('\n')
        }
      }
    }

    if (exportType == ExportType.PARALLEL_COMPOSABLE) {
      val ext = fs.getCodecExtension(filename)
      val headerPath = parallelOutputPath + "/header" + ext
      val headerOs = if (ext == ".bgz") {
        val os = fs.createNoCompression(headerPath)
        new ComposableBGzipOutputStream(os)
      } else {
        fs.create(headerPath)
      }
      using(new OutputStreamWriter(headerOs)) { out =>
        header.foreach { h =>
          out.write(h)
          out.write('\n')
        }
      }

      // this filename should sort after every partition
      using(new OutputStreamWriter(fs.create(parallelOutputPath + "/part-composable-end" + ext))) { out =>
        // do nothing, for bgzip, this will write the empty block
      }
    }

    if (!fs.isFile(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    if (exportType == ExportType.CONCATENATED) {
      fs.copyMerge(parallelOutputPath, filename, rWithHeader.getNumPartitions, header = false)
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

  def subsetPartitions(keep: IndexedSeq[Int], newPartitioner: Option[Partitioner] = None)(implicit ct: ClassTag[T]): RDD[T] = {
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
    oldToNewPI: IndexedSeq[Int],
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

  def writePartitions(
    ctx: ExecuteContext,
    path: String,
    stageLocally: Boolean,
    write: (Iterator[T], OutputStream) => (Long, Long)
  )(implicit tct: ClassTag[T]
  ): (Array[FileWriteMetadata]) =
    ContextRDD.weaken(r).writePartitions(ctx,
      path,
      null,
      stageLocally,
      (_, _) => null,
      (_, it, os, _) => write(it, os))
}

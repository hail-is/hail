package is.hail.sparkextras

import is.hail.asm4s.HailClassLoader
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.{unsafeHailClassLoaderForSparkWorkers, SparkBackend, SparkTaskContext}
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD.{inCtx, Element}
import is.hail.sparkextras.implicits.{toRichContextRDD, toRichRDD, toRichSC}
import is.hail.utils._

import scala.collection.compat._
import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.rdd._

object Combiner {
  def apply[U](zero: => U, combine: (U, U) => U, commutative: Boolean, associative: Boolean)
    : Combiner[U] = {
    assert(associative)
    if (commutative)
      new CommutativeAndAssociativeCombiner(zero, combine)
    else
      new AssociativeCombiner(zero, combine)
  }
}

abstract class Combiner[U] {
  def combine(i: Int, value0: U): Unit

  def result(): U
}

class CommutativeAndAssociativeCombiner[U](zero: => U, combine: (U, U) => U) extends Combiner[U] {
  var state: U = zero

  override def combine(i: Int, value0: U): Unit = state = combine(state, value0)

  override def result(): U = state
}

class AssociativeCombiner[U](zero: => U, combine: (U, U) => U) extends Combiner[U] with Logging {

  case class TreeValue(var value: U, var end: Int)

  // The type U may contain resources, e.g. regions. 't' has ownership of every
  // U it holds.
  private val t = new java.util.TreeMap[Int, TreeValue]()

  override def combine(i: Int, value0: U): Unit = {
    logger.info(s"at result $i, AssociativeCombiner contains ${t.size()} queued results")
    var value = value0
    var end = i

    val nexttv = t.get(i + 1)
    if (nexttv != null) {
      value = combine(value, nexttv.value)
      end = nexttv.end
      t.remove(i + 1)
    }

    val prevEntry = t.floorEntry(i - 1)
    if (prevEntry != null) {
      val prevtv = prevEntry.getValue
      if (prevtv.end == i - 1) {
        prevtv.value = combine(prevtv.value, value)
        prevtv.end = end
        return
      }
    }

    t.put(i, TreeValue(value, end))
  }

  override def result(): U = {
    // after 'result' returns, 't' owns no values.
    val n = t.size()
    if (n > 0) {
      assert(n == 1)
      t.firstEntry().getValue.value
    } else
      zero
  }
}

object ContextRDD {

  type Element[A] = (HailClassLoader, RVDContext) => Iterator[A]

  def inCtx[U](f: Element[U]): Iterator[Element[U]] =
    Iterator.single(f)

  def apply[T: ClassTag](rdd: RDD[Element[T]]): ContextRDD[T] =
    new ContextRDD(rdd)

  def empty[T: ClassTag](): ContextRDD[T] =
    new ContextRDD(SparkBackend.sparkContext.emptyRDD)

  def union[T: ClassTag](sc: SparkContext, xs: Seq[ContextRDD[T]]): ContextRDD[T] =
    new ContextRDD(sc.union(xs.map(_.rdd)))

  def weaken[T: ClassTag](rdd: RDD[T]): ContextRDD[T] =
    new ContextRDD(rdd.mapPartitions(it => inCtx((_, _) => it)))

  def textFilesLines(
    ctx: ExecuteContext,
    files: Array[String],
    nPartitions: Option[Int] = None,
    filterAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace(),
  ): ContextRDD[WithContext[String]] =
    textFilesLines(
      files,
      nPartitions.getOrElse(ctx.backend.defaultParallelism),
      filterAndReplace,
    )

  def textFilesLines(
    files: Array[String],
    nPartitions: Int,
    filterAndReplace: TextInputFilterAndReplace,
  ): ContextRDD[WithContext[String]] =
    ContextRDD.weaken(
      SparkBackend.sparkContext.textFilesLines(
        files,
        nPartitions,
      )
        .mapPartitions(filterAndReplace.apply)
    )

  def parallelize[T: ClassTag](sc: SparkContext, data: Seq[T], nPartitions: Option[Int] = None)
    : ContextRDD[T] =
    weaken(sc.parallelize(data, nPartitions.getOrElse(sc.defaultMinPartitions))).map(x => x)

  def parallelize[T: ClassTag](data: Seq[T], numSlices: Int): ContextRDD[T] =
    weaken(SparkBackend.sparkContext.parallelize(data, numSlices)).map {
      x => x
    }

  def parallelize[T: ClassTag](data: Seq[T]): ContextRDD[T] =
    weaken(SparkBackend.sparkContext.parallelize(data)).map(x => x)

  def czipNPartitions[T: ClassTag, U: ClassTag](
    crdds: IndexedSeq[ContextRDD[T]]
  )(
    f: (RVDContext, Array[Iterator[T]]) => Iterator[U]
  ): ContextRDD[U] =
    new ContextRDD(
      MultiWayZipPartitionsRDD(crdds.map(_.rdd)) { its =>
        inCtx((hcl, ctx) => f(ctx, its.map(_.flatMap(_(hcl, ctx)))))
      }
    )
}

class ContextRDD[T: ClassTag](val rdd: RDD[Element[T]]) extends Serializable {

  private[this] def sparkManagedContext[U](f: (HailClassLoader, RVDContext) => U): U = {
    val c = RVDContext.default(SparkTaskContext.get().getRegionPool())
    TaskContext.get().addTaskCompletionListener[Unit]((_: TaskContext) => c.close()): Unit
    f(unsafeHailClassLoaderForSparkWorkers, c)
  }

  def run[U >: T: ClassTag]: RDD[U] =
    this.cleanupRegions.rdd.mapPartitions { part =>
      sparkManagedContext((hcl, ctx) => part.flatMap(_(hcl, ctx)))
    }

  def collect(): Array[T] =
    run.collect()

  def map[U: ClassTag](f: T => U): ContextRDD[U] =
    mapPartitions(_.map(f), preservesPartitioning = true)

  def filter(f: T => Boolean): ContextRDD[T] =
    mapPartitions(_.filter(f), preservesPartitioning = true)

  def flatMap[U: ClassTag](f: T => IterableOnce[U]): ContextRDD[U] =
    mapPartitions(_.flatMap(f))

  def mapPartitions[U: ClassTag](
    f: Iterator[T] => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    cmapPartitions((_, _, part) => f(part), preservesPartitioning)

  def mapPartitionsWithIndex[U: ClassTag](
    f: (Int, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    cmapPartitionsWithIndex((i, _, _, part) => f(i, part), preservesPartitioning)

  def cmap[U: ClassTag](f: (HailClassLoader, RVDContext, T) => U): ContextRDD[U] =
    cmapPartitions((hcl, c, it) => it.map(f(hcl, c, _)), true)

  def cfilter(f: (HailClassLoader, RVDContext, T) => Boolean): ContextRDD[T] =
    cmapPartitions((hcl, c, it) => it.filter(f(hcl, c, _)), true)

  def cflatMap[U: ClassTag](f: (HailClassLoader, RVDContext, T) => IterableOnce[U]): ContextRDD[U] =
    cmapPartitions((hcl, c, it) => it.flatMap(f(hcl, c, _)))

  def cmapPartitions[U: ClassTag](
    f: (HailClassLoader, RVDContext, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] = new ContextRDD(
    rdd.mapPartitions(
      part => inCtx((hcl, ctx) => f(hcl, ctx, part.flatMap(_(hcl, ctx)))),
      preservesPartitioning,
    )
  )

  def cmapPartitionsWithContext[U: ClassTag](
    f: (HailClassLoader, RVDContext, Element[T]) => Iterator[U]
  ): ContextRDD[U] =
    cmapPartitionsWithContextAndIndex((_, hcl, ctx, x) => f(hcl, ctx, x))

  def cmapPartitionsWithContextAndIndex[U: ClassTag](
    f: (Int, HailClassLoader, RVDContext, Element[T]) => Iterator[U]
  ): ContextRDD[U] =
    new ContextRDD(rdd.mapPartitionsWithIndex { (i, part) =>
      part.flatMap(x => inCtx((hcl, ctx) => f(i, hcl, ctx, x)))
    })

  // Gives consumer ownership of the context. Consumer is responsible for freeing
  // resources per element.
  def crunJobWithIndex[U: ClassTag](f: (Int, HailClassLoader, RVDContext, Iterator[T]) => U)
    : Array[U] =
    sparkContext.runJob(
      rdd,
      { (taskContext, it: Iterator[Element[T]]) =>
        val c = RVDContext.default(SparkTaskContext.get().getRegionPool())
        val hcl = unsafeHailClassLoaderForSparkWorkers
        val ans = f(taskContext.partitionId(), hcl, c, it.flatMap(_(hcl, c)))
        ans
      },
    )

  def cmapPartitionsAndContext[U: ClassTag](
    f: (HailClassLoader, RVDContext, Iterator[Element[T]]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    onRDD(_.mapPartitions(
      part => inCtx((hcl, ctx) => f(hcl, ctx, part)),
      preservesPartitioning,
    ))

  def cmapPartitionsWithIndex[U: ClassTag](
    f: (Int, HailClassLoader, RVDContext, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] = new ContextRDD(
    rdd.mapPartitionsWithIndex(
      (i, part) => inCtx((hcl, ctx) => f(i, hcl, ctx, part.flatMap(_(hcl, ctx)))),
      preservesPartitioning,
    )
  )

  def cmapPartitionsWithIndexAndValue[U: ClassTag, V](
    values: Array[V],
    f: (Int, HailClassLoader, RVDContext, V, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] = new ContextRDD(
    new MapPartitionsWithValueRDD[Element[T], Element[U], V](
      rdd,
      values,
      (i, v, part) => inCtx((hcl, ctx) => f(i, hcl, ctx, v, part.flatMap(_(hcl, ctx)))),
      preservesPartitioning,
    )
  )

  def cmapPartitionsAndContextWithIndex[U: ClassTag](
    f: (Int, HailClassLoader, RVDContext, Iterator[Element[T]]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    onRDD(_.mapPartitionsWithIndex(
      (i, part) => inCtx((hcl, ctx) => f(i, hcl, ctx, part)),
      preservesPartitioning,
    ))

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def zipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[V] =
    czipPartitions[U, V](that, preservesPartitioning)((_, _, l, r) => f(l, r))

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def czipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (HailClassLoader, RVDContext, Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)((l, r) =>
      inCtx((hcl, ctx) => f(hcl, ctx, l.flatMap(_(hcl, ctx)), r.flatMap(_(hcl, ctx))))
    )
  )

  def subsetPartitions(keptPartitionIndices: Array[Int]): ContextRDD[T] =
    onRDD(_.subsetPartitions(keptPartitionIndices))

  def reorderPartitions(oldIndices: Array[Int]): ContextRDD[T] =
    onRDD(_.reorderPartitions(oldIndices))

  def noShuffleCoalesce(numPartitions: Int): ContextRDD[T] =
    onRDD(_.coalesce(numPartitions, false))

  def shuffleCoalesce(numPartitions: Int): ContextRDD[T] =
    ContextRDD.weaken(run.coalesce(numPartitions, true))

  // partEnds are the inclusive index of the last element of parts to be coalesced, that is, for
  // a ContextRDD with 8 partitions, being naively coalesced to 3, one example set of part ends is
  // [2, 5, 7]. With this, original partion indicies 0, 1, and 2 make up the first new partition 3,
  // 4, and 5 make up the second, and 6 and 7 make up the third.
  def coalesceWithEnds(partEnds: Array[Int]): ContextRDD[T] =
    onRDD { rdd =>
      rdd.coalesce(
        partEnds.length,
        shuffle = false,
        partitionCoalescer = Some(new CRDDCoalescer(partEnds)),
      )
    }

  def runJob[U: ClassTag](f: Iterator[T] => U, partitions: Seq[Int]): Array[U] =
    sparkContext.runJob(
      rdd,
      (it: Iterator[Element[T]]) => sparkManagedContext((hcl, c) => f(it.flatMap(_(hcl, c)))),
      partitions,
    )

  def blocked(partFirst: Array[Int], partLast: Array[Int]): ContextRDD[T] =
    new ContextRDD(new BlockedRDD(rdd, partFirst, partLast))

  def sparkContext: SparkContext = rdd.sparkContext

  def getNumPartitions: Int = rdd.getNumPartitions

  def preferredLocations(partition: Partition): Seq[String] =
    rdd.preferredLocations(partition)

  def partitions: Array[Partition] = rdd.partitions

  def partitioner: Option[Partitioner] = rdd.partitioner

  def iterator(p: Partition, tc: TaskContext): Iterator[Element[T]] =
    rdd.iterator(p, tc)

  def iterator(p: Partition, tc: TaskContext, hcl: HailClassLoader, ctx: RVDContext): Iterator[T] =
    rdd.iterator(p, tc).flatMap(_(hcl, ctx))

  private[this] def onRDD[U: ClassTag](f: RDD[Element[T]] => RDD[Element[U]]): ContextRDD[U] =
    new ContextRDD(f(rdd))
}

private class CRDDCoalescer(partEnds: Array[Int]) extends PartitionCoalescer with Serializable {
  override def coalesce(maxPartitions: Int, prev: RDD[_]): Array[PartitionGroup] = {
    assert(maxPartitions == partEnds.length)
    val groups = Array.fill(maxPartitions)(new PartitionGroup())
    val parts = prev.partitions
    var i = 0
    for ((end, j) <- partEnds.zipWithIndex)
      while (i <= end) {
        groups(j).partitions += parts(i)
        i += 1
      }
    groups
  }
}

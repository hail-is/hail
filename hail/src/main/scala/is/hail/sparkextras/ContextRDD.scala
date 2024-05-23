package is.hail.sparkextras

import is.hail.HailContext
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.rvd.RVDContext
import is.hail.utils._

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

  def combine(i: Int, value0: U): Unit = state = combine(state, value0)

  def result(): U = state
}

class AssociativeCombiner[U](zero: => U, combine: (U, U) => U) extends Combiner[U] {

  case class TreeValue(var value: U, var end: Int)

  // The type U may contain resources, e.g. regions. 't' has ownership of every
  // U it holds.
  private val t = new java.util.TreeMap[Int, TreeValue]()

  def combine(i: Int, value0: U): Unit = {
    log.info(s"at result $i, AssociativeCombiner contains ${t.size()} queued results")
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

  def result(): U = {
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
  def apply[T: ClassTag](
    rdd: RDD[RVDContext => Iterator[T]]
  ): ContextRDD[T] = new ContextRDD(rdd)

  def empty[T: ClassTag](): ContextRDD[T] =
    new ContextRDD(
      SparkBackend.sparkContext("ContextRDD.empty").emptyRDD[RVDContext => Iterator[T]]
    )

  def union[T: ClassTag](
    sc: SparkContext,
    xs: Seq[ContextRDD[T]],
  ): ContextRDD[T] =
    new ContextRDD(sc.union(xs.map(_.rdd)))

  def weaken[T: ClassTag](
    rdd: RDD[T]
  ): ContextRDD[T] =
    new ContextRDD(rdd.mapPartitions(it => Iterator.single((ctx: RVDContext) => it)))

  def textFilesLines(
    files: Array[String],
    nPartitions: Option[Int] = None,
    filterAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace(),
  ): ContextRDD[WithContext[String]] =
    textFilesLines(
      files,
      nPartitions.getOrElse(HailContext.backend.defaultParallelism),
      filterAndReplace,
    )

  def textFilesLines(
    files: Array[String],
    nPartitions: Int,
    filterAndReplace: TextInputFilterAndReplace,
  ): ContextRDD[WithContext[String]] =
    ContextRDD.weaken(
      SparkBackend.sparkContext("ContxtRDD.textFilesLines").textFilesLines(
        files,
        nPartitions,
      )
        .mapPartitions(filterAndReplace.apply)
    )

  def parallelize[T: ClassTag](sc: SparkContext, data: Seq[T], nPartitions: Option[Int] = None)
    : ContextRDD[T] =
    weaken(sc.parallelize(data, nPartitions.getOrElse(sc.defaultMinPartitions))).map(x => x)

  def parallelize[T: ClassTag](data: Seq[T], numSlices: Int): ContextRDD[T] =
    weaken(SparkBackend.sparkContext("ContextRDD.parallelize").parallelize(data, numSlices)).map {
      x => x
    }

  def parallelize[T: ClassTag](data: Seq[T]): ContextRDD[T] =
    weaken(SparkBackend.sparkContext("ContextRDD.parallelize").parallelize(data)).map(x => x)

  type ElementType[T] = RVDContext => Iterator[T]

  def czipNPartitions[T: ClassTag, U: ClassTag](
    crdds: IndexedSeq[ContextRDD[T]],
    preservesPartitioning: Boolean = false,
  )(
    f: (RVDContext, Array[Iterator[T]]) => Iterator[U]
  ): ContextRDD[U] = {
    def inCtx(f: RVDContext => Iterator[U]): Iterator[RVDContext => Iterator[U]] =
      Iterator.single(f)
    new ContextRDD(
      MultiWayZipPartitionsRDD(crdds.map(_.rdd)) { its =>
        inCtx(ctx => f(ctx, its.map(_.flatMap(_(ctx)))))
      }
    )
  }
}

class ContextRDD[T: ClassTag](
  val rdd: RDD[RVDContext => Iterator[T]]
) extends Serializable {
  type ElementType = ContextRDD.ElementType[T]

  private[this] def sparkManagedContext[U](func: RVDContext => U): U = {
    val c = RVDContext.default(SparkTaskContext.get().getRegionPool())
    TaskContext.get().addTaskCompletionListener[Unit]((_: TaskContext) => c.close())
    func(c)
  }

  def run[U >: T: ClassTag]: RDD[U] =
    this.cleanupRegions.rdd.mapPartitions(part => sparkManagedContext(c => part.flatMap(_(c))))

  def collect(): Array[T] =
    run.collect()

  private[this] def inCtx[U: ClassTag](
    f: RVDContext => Iterator[U]
  ): Iterator[RVDContext => Iterator[U]] = Iterator.single(f)

  def map[U: ClassTag](f: T => U): ContextRDD[U] =
    mapPartitions(_.map(f), preservesPartitioning = true)

  def filter(f: T => Boolean): ContextRDD[T] =
    mapPartitions(_.filter(f), preservesPartitioning = true)

  def flatMap[U: ClassTag](f: T => TraversableOnce[U]): ContextRDD[U] =
    mapPartitions(_.flatMap(f))

  def mapPartitions[U: ClassTag](
    f: Iterator[T] => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    cmapPartitions((_, part) => f(part), preservesPartitioning)

  def mapPartitionsWithIndex[U: ClassTag](
    f: (Int, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    cmapPartitionsWithIndex((i, _, part) => f(i, part), preservesPartitioning)

  def cmap[U: ClassTag](f: (RVDContext, T) => U): ContextRDD[U] =
    cmapPartitions((c, it) => it.map(f(c, _)), true)

  def cfilter(f: (RVDContext, T) => Boolean): ContextRDD[T] =
    cmapPartitions((c, it) => it.filter(f(c, _)), true)

  def cflatMap[U: ClassTag](f: (RVDContext, T) => TraversableOnce[U]): ContextRDD[U] =
    cmapPartitions((c, it) => it.flatMap(f(c, _)))

  def cmapPartitions[U: ClassTag](
    f: (RVDContext, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] = new ContextRDD(
    rdd.mapPartitions(
      part => inCtx(ctx => f(ctx, part.flatMap(_(ctx)))),
      preservesPartitioning,
    )
  )

  def cmapPartitionsWithContext[U: ClassTag](
    f: (RVDContext, (RVDContext) => Iterator[T]) => Iterator[U]
  ): ContextRDD[U] =
    new ContextRDD(rdd.mapPartitions(part =>
      part.flatMap {
        x => inCtx(consumerCtx => f(consumerCtx, x))
      }
    ))

  def cmapPartitionsWithContextAndIndex[U: ClassTag](
    f: (Int, RVDContext, (RVDContext) => Iterator[T]) => Iterator[U]
  ): ContextRDD[U] =
    new ContextRDD(rdd.mapPartitionsWithIndex((i, part) =>
      part.flatMap {
        x => inCtx(consumerCtx => f(i, consumerCtx, x))
      }
    ))

  // Gives consumer ownership of the context. Consumer is responsible for freeing
  // resources per element.
  def crunJobWithIndex[U: ClassTag](f: (Int, RVDContext, Iterator[T]) => U): Array[U] =
    sparkContext.runJob(
      rdd,
      { (taskContext, it: Iterator[RVDContext => Iterator[T]]) =>
        val c = RVDContext.default(SparkTaskContext.get().getRegionPool())
        val ans = f(taskContext.partitionId(), c, it.flatMap(_(c)))
        ans
      },
    )

  def cmapPartitionsAndContext[U: ClassTag](
    f: (RVDContext, (Iterator[RVDContext => Iterator[T]])) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    onRDD(_.mapPartitions(
      part => inCtx(ctx => f(ctx, part)),
      preservesPartitioning,
    ))

  def cmapPartitionsWithIndex[U: ClassTag](
    f: (Int, RVDContext, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] = new ContextRDD(
    rdd.mapPartitionsWithIndex(
      (i, part) => inCtx(ctx => f(i, ctx, part.flatMap(_(ctx)))),
      preservesPartitioning,
    )
  )

  def cmapPartitionsWithIndexAndValue[U: ClassTag, V](
    values: Array[V],
    f: (Int, RVDContext, V, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] = new ContextRDD(
    new MapPartitionsWithValueRDD[(RVDContext) => Iterator[T], (RVDContext) => Iterator[U], V](
      rdd,
      values,
      (i, v, part) => inCtx(ctx => f(i, ctx, v, part.flatMap(_(ctx)))),
      preservesPartitioning,
    )
  )

  def cmapPartitionsAndContextWithIndex[U: ClassTag](
    f: (Int, RVDContext, Iterator[RVDContext => Iterator[T]]) => Iterator[U],
    preservesPartitioning: Boolean = false,
  ): ContextRDD[U] =
    onRDD(_.mapPartitionsWithIndex(
      (i, part) => inCtx(ctx => f(i, ctx, part)),
      preservesPartitioning,
    ))

  def czip[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (RVDContext, T, U) => V
  ): ContextRDD[V] = czipPartitions(that, preservesPartitioning) { (ctx, l, r) =>
    new Iterator[V] {
      def hasNext = {
        val lhn = l.hasNext
        val rhn = r.hasNext
        assert(lhn == rhn)
        lhn
      }
      def next(): V =
        f(ctx, l.next(), r.next())
    }
  }

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def zipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[V] =
    czipPartitions[U, V](that, preservesPartitioning)((_, l, r) => f(l, r))

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def czipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (RVDContext, Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)((l, r) =>
      inCtx(ctx => f(ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))
    )
  )

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def czipPartitionsWithIndex[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (Int, RVDContext, Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)((l, r) =>
      Iterator.single(l -> r)
    ).mapPartitionsWithIndex(
      { case (i, it) =>
        it.flatMap { case (l, r) =>
          inCtx(ctx => f(i, ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))
        }
      },
      preservesPartitioning,
    )
  )

  def czipPartitionsAndContext[U: ClassTag, V: ClassTag](
    that: ContextRDD[U],
    preservesPartitioning: Boolean = false,
  )(
    f: (
      RVDContext,
      Iterator[RVDContext => Iterator[T]],
      Iterator[RVDContext => Iterator[U]],
    ) => Iterator[V]
  ): ContextRDD[V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)((l, r) => inCtx(ctx => f(ctx, l, r)))
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
      (it: Iterator[ElementType]) => sparkManagedContext(c => f(it.flatMap(_(c)))),
      partitions,
    )

  def blocked(partFirst: Array[Int], partLast: Array[Int]): ContextRDD[T] =
    new ContextRDD(new BlockedRDD(rdd, partFirst, partLast))

  def sparkContext: SparkContext = rdd.sparkContext

  def getNumPartitions: Int = rdd.getNumPartitions

  def preferredLocations(partition: Partition): Seq[String] =
    rdd.preferredLocations(partition)

  private[this] def clean[U <: AnyRef](value: U): U =
    ExposedUtils.clean(value)

  def partitions: Array[Partition] = rdd.partitions

  def partitioner: Option[Partitioner] = rdd.partitioner

  def iterator(p: Partition, tc: TaskContext): Iterator[ElementType] =
    rdd.iterator(p, tc)

  def iterator(p: Partition, tc: TaskContext, ctx: RVDContext): Iterator[T] =
    rdd.iterator(p, tc).flatMap(_(ctx))

  private[this] def onRDD[U: ClassTag](
    f: RDD[RVDContext => Iterator[T]] => RDD[RVDContext => Iterator[U]]
  ): ContextRDD[U] = new ContextRDD(f(rdd))
}

private class CRDDCoalescer(partEnds: Array[Int]) extends PartitionCoalescer with Serializable {
  def coalesce(maxPartitions: Int, prev: RDD[_]): Array[PartitionGroup] = {
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

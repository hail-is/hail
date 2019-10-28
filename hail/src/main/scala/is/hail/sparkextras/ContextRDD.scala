package is.hail.sparkextras

import is.hail.utils._
import is.hail.utils.PartitionCounts._
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.ExposedUtils

import scala.reflect.ClassTag

object Combiner {
  def apply[U](zero: U, combine: (U, U) => U, commutative: Boolean, associative: Boolean): Combiner[U] = {
    assert(associative)
    if (commutative)
      new CommutativeAndAssociativeCombiner(zero, combine)
    else
      new AssociativeCombiner(zero, combine)
  }
}

abstract class Combiner[U] {
  def combine(i: Int, value0: U)

  def result(): U
}

class CommutativeAndAssociativeCombiner[U](zero: U, combine: (U, U) => U) extends Combiner[U] {
  var state: U = zero

  def combine(i: Int, value0: U): Unit = state = combine(state, value0)

  def result(): U = state
}

class AssociativeCombiner[U](zero: U, combine: (U, U) => U) extends Combiner[U] {

  case class TreeValue(var value: U, var end: Int)

  private val t = new java.util.TreeMap[Int, TreeValue]()

  def combine(i: Int, value0: U) {
    log.info(s"at result $i, AssociativeCombiner contains ${ t.size() } queued results")
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
    val n = t.size()
    if (n > 0) {
      assert(n == 1)
      t.firstEntry().getValue.value
    } else
      zero
  }
}

object ContextRDD {
  def apply[C <: AutoCloseable : Pointed, T: ClassTag](
    rdd: RDD[C => Iterator[T]]
  ): ContextRDD[C, T] = new ContextRDD(rdd, () => point[C]())

  def empty[C <: AutoCloseable, T: ClassTag](
    sc: SparkContext,
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(sc.emptyRDD[C => Iterator[T]], mkc)

  def empty[C <: AutoCloseable : Pointed, T: ClassTag](
    sc: SparkContext
  ): ContextRDD[C, T] =
    new ContextRDD(sc.emptyRDD[C => Iterator[T]], () => point[C]())

  // this one weird trick permits the caller to specify T without C
  sealed trait Empty[T] {
    def apply[C <: AutoCloseable](
      sc: SparkContext,
      mkc: () => C
    )(implicit tct: ClassTag[T]
    ): ContextRDD[C, T] = empty(sc, mkc)
  }
  private[this] object emptyInstance extends Empty[Nothing]
  def empty[T] = emptyInstance.asInstanceOf[Empty[T]]

  def union[C <: AutoCloseable : Pointed, T: ClassTag](
    sc: SparkContext,
    xs: Seq[ContextRDD[C, T]]
  ): ContextRDD[C, T] = union(sc, xs, () => point[C]())

  def union[C <: AutoCloseable, T: ClassTag](
    sc: SparkContext,
    xs: Seq[ContextRDD[C, T]],
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(sc.union(xs.map(_.rdd)), mkc)

  def weaken[C <: AutoCloseable, T: ClassTag](
    rdd: RDD[T],
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(rdd.mapPartitions(it => Iterator.single((ctx: C) => it)), mkc)

  // this one weird trick permits the caller to specify C without T
  sealed trait Weaken[C <: AutoCloseable] {
    def apply[T: ClassTag](
      rdd: RDD[T]
    )(implicit c: Pointed[C]
    ): ContextRDD[C, T] = weaken(rdd, c.point _)
  }
  private[this] object weakenInstance extends Weaken[Nothing]
  def weaken[C <: AutoCloseable] = weakenInstance.asInstanceOf[Weaken[C]]

  def textFilesLines[C <: AutoCloseable](
    sc: SparkContext,
    files: Array[String],
    nPartitions: Option[Int] = None,
    filterAndReplace: TextInputFilterAndReplace = TextInputFilterAndReplace()
  )(implicit c: Pointed[C]
  ): ContextRDD[C, WithContext[String]] =
    textFilesLines(
      sc,
      files,
      nPartitions.getOrElse(sc.defaultMinPartitions),
      filterAndReplace)

  def textFilesLines[C <: AutoCloseable](
    sc: SparkContext,
    files: Array[String],
    nPartitions: Int,
    filterAndReplace: TextInputFilterAndReplace
  )(implicit c: Pointed[C]
  ): ContextRDD[C, WithContext[String]] =
    ContextRDD.weaken[C](
      sc.textFilesLines(
        files,
        nPartitions)
        .mapPartitions(filterAndReplace.apply))

  // this one weird trick permits the caller to specify C without T
  sealed trait Parallelize[C <: AutoCloseable] {
    def apply[T : ClassTag](
      sc: SparkContext,
      data: Seq[T],
      nPartitions: Option[Int] = None
    )(implicit c: Pointed[C]
    ): ContextRDD[C, T] = ContextRDD.weaken[C](
      sc.parallelize(
        data,
        nPartitions.getOrElse(sc.defaultMinPartitions)))

    def apply[T : ClassTag](
      sc: SparkContext,
      data: Seq[T],
      numSlices: Int
    )(implicit c: Pointed[C]
    ): ContextRDD[C, T] = weaken(sc.parallelize(data, numSlices))

    def apply[T : ClassTag](
      sc: SparkContext,
      data: Seq[T]
    )(implicit c: Pointed[C]
    ): ContextRDD[C, T] = weaken(sc.parallelize(data))
  }
  private[this] object parallelizeInstance extends Parallelize[Nothing]
  def parallelize[C <: AutoCloseable] = parallelizeInstance.asInstanceOf[Parallelize[C]]

  type ElementType[C, T] = C => Iterator[T]

  def czipNPartitions[C <: AutoCloseable, T: ClassTag, U: ClassTag](
    crdds: IndexedSeq[ContextRDD[C, T]],
    preservesPartitioning: Boolean = false
  )(f: (C, Array[Iterator[T]]) => Iterator[U]
  ): ContextRDD[C, U] = {
    val mkc = crdds.head.mkc
    def inCtx(f: C => Iterator[U]): Iterator[C => Iterator[U]] = Iterator.single(f)
    new ContextRDD(
      MultiWayZipPartitionsRDD(crdds.map(_.rdd)) { its =>
        inCtx(ctx => f(ctx, its.map(_.flatMap(_(ctx)))))
      },
      mkc)
  }
}

class ContextRDD[C <: AutoCloseable, T: ClassTag](
  val rdd: RDD[C => Iterator[T]],
  val mkc: () => C
) extends Serializable {
  type ElementType = ContextRDD.ElementType[C, T]

  private[this] def sparkManagedContext(): C = {
    val c = mkc()
    TaskContext.get().addTaskCompletionListener { (_: TaskContext) =>
      c.close()
    }
    c
  }

  def run[U >: T : ClassTag]: RDD[U] =
    rdd.mapPartitions { part =>
      val c = sparkManagedContext()
      part.flatMap(_(c))
    }

  def collect(): Array[T] =
    run.collect()

  private[this] def inCtx[U: ClassTag](
    f: C => Iterator[U]
  ): Iterator[C => Iterator[U]] = Iterator.single(f)

  def map[U: ClassTag](f: T => U): ContextRDD[C, U] =
    mapPartitions(_.map(f), preservesPartitioning = true)

  def filter(f: T => Boolean): ContextRDD[C, T] =
    mapPartitions(_.filter(f), preservesPartitioning = true)

  def flatMap[U: ClassTag](f: T => TraversableOnce[U]): ContextRDD[C, U] =
    mapPartitions(_.flatMap(f))

  def mapPartitions[U: ClassTag](
    f: Iterator[T] => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] =
    cmapPartitions((_, part) => f(part), preservesPartitioning)

  def mapPartitionsWithIndex[U: ClassTag](
    f: (Int, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] =
    cmapPartitionsWithIndex((i, _, part) => f(i, part), preservesPartitioning)

  def cmap[U: ClassTag](f: (C, T) => U): ContextRDD[C, U] =
    cmapPartitions((c, it) => it.map(f(c, _)), true)

  def cfilter(f: (C, T) => Boolean): ContextRDD[C, T] =
    cmapPartitions((c, it) => it.filter(f(c, _)), true)

  def cflatMap[U: ClassTag](f: (C, T) => TraversableOnce[U]): ContextRDD[C, U] =
    cmapPartitions((c, it) => it.flatMap(f(c, _)))

  def cmapPartitions[U: ClassTag](
    f: (C, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] = new ContextRDD(
    rdd.mapPartitions(
      part => inCtx(ctx => f(ctx, part.flatMap(_(ctx)))),
      preservesPartitioning),
    mkc)

  def cmapPartitionsAndContext[U: ClassTag](
    f: (C, (Iterator[C => Iterator[T]])) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] =
    onRDD(_.mapPartitions(
      part => inCtx(ctx => f(ctx, part)),
      preservesPartitioning))

  def cmapPartitionsWithIndex[U: ClassTag](
    f: (Int, C, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] = new ContextRDD(
    rdd.mapPartitionsWithIndex(
      (i, part) => inCtx(ctx => f(i, ctx, part.flatMap(_(ctx)))),
      preservesPartitioning),
    mkc)

  def cmapPartitionsWithIndexAndValue[U: ClassTag, V](
    values: Array[V],
    f: (Int, C, V, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] = new ContextRDD(
    new MapPartitionsWithValueRDD[(C) => Iterator[T], (C) => Iterator[U], V](
      rdd,
      values,
      (i, v, part) => inCtx(ctx => f(i, ctx, v, part.flatMap(_(ctx)))),
      preservesPartitioning),
    mkc)

  def cmapPartitionsAndContextWithIndex[U: ClassTag](
    f: (Int, C, Iterator[C => Iterator[T]]) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] =
    onRDD(_.mapPartitionsWithIndex(
      (i, part) => inCtx(ctx => f(i, ctx, part)),
      preservesPartitioning))

  def czip[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (C, T, U) => V
  ): ContextRDD[C, V] = czipPartitions(that, preservesPartitioning) { (ctx, l, r) =>
    new Iterator[V] {
      def hasNext = {
        val lhn = l.hasNext
        val rhn = r.hasNext
        assert(lhn == rhn)
        lhn
      }
      def next(): V = {
        f(ctx, l.next(), r.next())
      }
    }
  }

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def zipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[C, V] =
    czipPartitions[U, V](that, preservesPartitioning)((_, l, r) => f(l, r))

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def czipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (C, Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[C, V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)(
      (l, r) => inCtx(ctx => f(ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))),
    mkc)

  // WARNING: this method is easy to use wrong because it shares the context
  // between the two producers and the one consumer
  def czipPartitionsWithIndex[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (Int, C, Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[C, V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)(
      (l, r) => Iterator.single(l -> r)).mapPartitionsWithIndex({ case (i, it) =>
      it.flatMap { case (l, r) =>
        inCtx(ctx => f(i, ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))
      }
    }, preservesPartitioning),
    mkc)

  def czipPartitionsAndContext[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (C, Iterator[C => Iterator[T]], Iterator[C => Iterator[U]]) => Iterator[V]
  ): ContextRDD[C, V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)(
      (l, r) => inCtx(ctx => f(ctx, l, r))),
    mkc)

  def subsetPartitions(keptPartitionIndices: Array[Int]): ContextRDD[C, T] =
    onRDD(_.subsetPartitions(keptPartitionIndices))

  def reorderPartitions(oldIndices: Array[Int]): ContextRDD[C, T] =
    onRDD(_.reorderPartitions(oldIndices))

  def noShuffleCoalesce(numPartitions: Int): ContextRDD[C, T] =
    onRDD(_.coalesce(numPartitions, false))

  def shuffleCoalesce(numPartitions: Int): ContextRDD[C, T] =
    ContextRDD.weaken(run.coalesce(numPartitions, true), mkc)

  def head(n: Long, partitionCounts: Option[IndexedSeq[Long]]): ContextRDD[C, T] = {
    require(n >= 0)

    val (idxLast, nTake) = partitionCounts match {
      case Some(pcs) =>
        getPCSubsetOffset(n, pcs.iterator) match {
          case Some(PCSubsetOffset(idx, nTake, _)) => idx -> nTake
          case None => return this
        }
      case None =>
        val PCSubsetOffset(idx, nTake, _) =
          incrementalPCSubsetOffset(n, 0 until getNumPartitions)(
            runJob(getIteratorSize, _)
          )
        idx -> nTake
    }

    mapPartitionsWithIndex({ case (i, it) =>
      if (i == idxLast)
        it.take(nTake.toInt)
      else
        it
    }, preservesPartitioning = true)
      .subsetPartitions((0 to idxLast).toArray)
  }

  def tail(n: Long, partitionCounts: Option[IndexedSeq[Long]]): ContextRDD[C, T] = {
    require(n >= 0)

    val (idxFirst, nDrop) = partitionCounts match {
      case Some(pcs) =>
        getPCSubsetOffset(n, pcs.reverseIterator) match {
          case Some(PCSubsetOffset(idx, _, nDrop)) => (pcs.length - idx - 1) -> nDrop
          case None => return this
        }
      case None =>
        val PCSubsetOffset(idx, _, nDrop) =
          incrementalPCSubsetOffset(n, Range.inclusive(getNumPartitions - 1, 0, -1))(
            runJob(getIteratorSize, _)
          )
        idx -> nDrop
    }

    mapPartitionsWithIndex({ case (i, it) =>
      if (i == idxFirst)
        it.drop(nDrop.toInt)
      else
        it
    }, preservesPartitioning = true)
      .subsetPartitions(Array.range(idxFirst, getNumPartitions))
  }

  def runJob[U: ClassTag](f: Iterator[T] => U, partitions: Seq[Int]): Array[U] =
    sparkContext.runJob(
      rdd,
      { (it: Iterator[ElementType]) =>
        val c = sparkManagedContext()
        f(it.flatMap(_(c)))
      },
      partitions)

  def blocked(partFirst: Array[Int], partLast: Array[Int]): ContextRDD[C, T] = {
    new ContextRDD(new BlockedRDD(rdd, partFirst, partLast), mkc)
  }

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

  def iterator(p: Partition, tc: TaskContext, ctx: C): Iterator[T] =
    rdd.iterator(p, tc).flatMap(_(ctx))

  private[this] def onRDD[U: ClassTag](
    f: RDD[C => Iterator[T]] => RDD[C => Iterator[U]]
  ): ContextRDD[C, U] = new ContextRDD(f(rdd), mkc)
}

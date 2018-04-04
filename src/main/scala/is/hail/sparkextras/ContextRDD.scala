package is.hail.sparkextras

import is.hail.utils._
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.storage._
import org.apache.spark.util.random._
import org.apache.spark.util.Utils

import scala.reflect.ClassTag

object ContextRDD {
  def apply[C <: ResettableContext : Pointed, T: ClassTag](
    rdd: RDD[C => Iterator[T]]
  ): ContextRDD[C, T] = new ContextRDD(rdd, point[C])

  def empty[C <: ResettableContext, T: ClassTag](
    sc: SparkContext,
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(sc.emptyRDD[C => Iterator[T]], mkc)

  def empty[C <: ResettableContext : Pointed, T: ClassTag](
    sc: SparkContext
  ): ContextRDD[C, T] =
    new ContextRDD(sc.emptyRDD[C => Iterator[T]], point[C])

  // this one weird trick permits the caller to specify T without C
  sealed trait Empty[T] {
    def apply[C <: ResettableContext](
      sc: SparkContext,
      mkc: () => C
    )(implicit tct: ClassTag[T]
    ): ContextRDD[C, T] = empty(sc, mkc)
  }
  private[this] object emptyInstance extends Empty[Nothing]
  def empty[T] = emptyInstance.asInstanceOf[Empty[T]]

  def union[C <: ResettableContext : Pointed, T: ClassTag](
    sc: SparkContext,
    xs: Seq[ContextRDD[C, T]]
  ): ContextRDD[C, T] = union(sc, xs, point[C])

  def union[C <: ResettableContext, T: ClassTag](
    sc: SparkContext,
    xs: Seq[ContextRDD[C, T]],
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(sc.union(xs.map(_.rdd)), mkc)

  def weaken[C <: ResettableContext, T: ClassTag](
    rdd: RDD[T],
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(rdd.mapPartitions(it => Iterator.single((ctx: C) => it)), mkc)

  // this one weird trick permits the caller to specify C without T
  sealed trait Weaken[C <: ResettableContext] {
    def apply[T : ClassTag](
      rdd: RDD[T]
    )(implicit c: Pointed[C]
    ): ContextRDD[C, T] = weaken(rdd, c.point _)
  }
  private[this] object weakenInstance extends Weaken[Nothing]
  def weaken[C <: ResettableContext] = weakenInstance.asInstanceOf[Weaken[C]]

  def textFilesLines[C <: ResettableContext](
      sc: SparkContext,
      files: Array[String],
      nPartitions: Option[Int] = None
    )(implicit c: Pointed[C]
    ): ContextRDD[C, WithContext[String]] =
      ContextRDD.weaken[C](
        sc.textFilesLines(
          files,
          nPartitions.getOrElse(sc.defaultMinPartitions)))


  // this one weird trick permits the caller to specify C without T
  sealed trait Parallelize[C <: ResettableContext] {
    def apply[T : ClassTag](
      sc: SparkContext,
      data: Seq[T],
      nPartitions: Option[Int] = None
    )(implicit c: Pointed[C]
    ): ContextRDD[C, T] = ContextRDD.weaken[C](
      sc.parallelize(
        data,
        nPartitions.getOrElse(sc.defaultMinPartitions)))
  }
  private[this] object parallelizeInstance extends Parallelize[Nothing]
  def parallelize[C <: ResettableContext] = parallelizeInstance.asInstanceOf[Parallelize[C]]

  type ElementType[C, T] = C => Iterator[T]
}

class ContextRDD[C <: ResettableContext, T: ClassTag](
  val rdd: RDD[C => Iterator[T]],
  val mkc: () => C
) extends Serializable {
  type ElementType = ContextRDD.ElementType[C, T]

  // WARNING: this resets the context, when this method is called, the value of
  // type `T` must already be "stable" i.e. not dependent on the region
  private[this] def decontextualize(c: C, it: Iterator[C => Iterator[T]]): Iterator[T] =
    it.flatMap { useCtx =>
      useCtx(c).map { t =>
        c.reset()
        t
      }
    }

  def run[U >: T : ClassTag]: RDD[U] =
    rdd.mapPartitions { part => using(mkc()) { cc => decontextualize(cc, part) } }

  private[this] def inCtx[U: ClassTag](
    f: C => Iterator[U]
  ): Iterator[C => Iterator[U]] = Iterator.single(f)

  private[this] def withoutContext[U: ClassTag](
    f: Iterator[T] => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] = cmapPartitions((_, it) => f(it), preservesPartitioning)

  def map[U: ClassTag](f: T => U): ContextRDD[C, U] =
    withoutContext(_.map(f), true)

  def filter(f: T => Boolean): ContextRDD[C, T] =
    withoutContext(_.filter(f), true)

  def flatMap[U: ClassTag](f: T => TraversableOnce[U]): ContextRDD[C, U] =
    withoutContext(_.flatMap(f))

  def mapPartitions[U: ClassTag](
    f: Iterator[T] => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] =
    cmapPartitions((_, part) => f(part), preservesPartitioning)

  def mapPartitionsWithIndex[U: ClassTag](
    f: (Int, Iterator[T]) => Iterator[U]
  ): ContextRDD[C, U] = cmapPartitionsWithIndex((i, _, part) => f(i, part))

  // FIXME: not safe because it uses same region for both
  // but clears region for each call to f's return value's next
  def zipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[C, V] =
    czipPartitions[U, V](that, preservesPartitioning)((_, l, r) => f(l, r))

  // FIXME: this should disappear when region values become non-serializable
  def aggregate[U: ClassTag](
    makeZero: () => U,
    seqOp: (U, T) => U,
    combOp: (U, U) => U
  ): U = aggregate[U, U](makeZero, seqOp, combOp, x => x, x => x)

  // FIXME: spark uses spark serialization to clone the zero, but its private,
  // so I just force the user to give me a constructor instead
  //
  // FIXME: do serialize and deserialize belong on contextrdd? They seem
  // necessary for the RegionValues as offsets to work, but it seems irrelevant
  // to ContextRDD
  def aggregate[U: ClassTag, V: ClassTag](
    makeZero: () => U,
    seqOp: (U, T) => U,
    combOp: (U, U) => U,
    serialize: U => V,
    deserialize: V => U
  ): U = {
    val zeroValue = makeZero()
    // FIXME: I need to clean seqOp and combOp but that's private in
    // SparkContext, wtf...
    val aggregatePartition = { (it: Iterator[C => Iterator[T]]) =>
      using(mkc()) { c =>
        serialize(it.flatMap(_(c)).aggregate(zeroValue)({ (t, u) =>
          val u2 = seqOp(t, u)
          c.reset()
          u2
        }, combOp)) } }
    var result = makeZero()
    val localCombiner = { (_: Int, v: V) =>
      result = combOp(result, deserialize(v)) }
    sparkContext.runJob(rdd, aggregatePartition, localCombiner)
    result
  }

  def treeAggregate[U: ClassTag](
    makeZero: () => U,
    seqOp: (U, T) => U,
    combOp: (U, U) => U
  ): U = treeAggregate(makeZero, seqOp, combOp, 2)

  def treeAggregate[U: ClassTag](
    makeZero: () => U,
    seqOp: (U, T) => U,
    combOp: (U, U) => U,
    depth: Int
  ): U = treeAggregate[U, U](makeZero, seqOp, combOp, (x: U) => x, (x: U) => x, depth)

  def treeAggregate[U: ClassTag, V: ClassTag](
    makeZero: () => U,
    seqOp: (U, T) => U,
    combOp: (U, U) => U,
    serialize: U => V,
    deserialize: V => U
  ): V = treeAggregate(makeZero, seqOp, combOp, serialize, deserialize, 2)

  def treeAggregate[U: ClassTag, V: ClassTag](
    makeZero: () => U,
    seqOp: (U, T) => U,
    combOp: (U, U) => U,
    serialize: U => V,
    deserialize: V => U,
    depth: Int
  ): V = {
    require(depth > 0)
    val zeroValue = serialize(makeZero())
    // FIXME: I need to clean seqOp and combOp but that's private in
    // SparkContext, wtf...
    val aggregatePartitionOfContextTs = { (it: Iterator[C => Iterator[T]]) =>
      using(mkc()) { c =>
        serialize(
          it.flatMap(_(c)).aggregate(deserialize(zeroValue))({ (t, u) =>
            val u2 = seqOp(t, u)
            c.reset()
            u2
          }, combOp)) } }
    val aggregatePartitionOfVs = { (it: Iterator[V]) =>
      using(mkc()) { c =>
        serialize(
          it.map(deserialize).fold(deserialize(zeroValue))(combOp)) } }
    val combOpV = { (l: V, r: V) =>
      using(mkc()) { c =>
        serialize(
          combOp(deserialize(l), deserialize(r))) } }

    var reduced: RDD[V] =
      rdd.mapPartitions(
        aggregatePartitionOfContextTs.andThen(Iterator.single _))
    var level = depth
    val scale =
      math.max(
        math.ceil(math.pow(reduced.partitions.length, 1.0 / depth)).toInt,
        2)
    var targetPartitionCount = reduced.partitions.length / scale

    while (level > 1 && targetPartitionCount >= scale) {
      reduced = reduced.mapPartitionsWithIndex { (i, it) =>
        it.map(i % targetPartitionCount -> _)
      }.reduceByKey(combOpV, targetPartitionCount).map(_._2)
      level -= 1
      targetPartitionCount /= scale
    }

    reduced.reduce(combOpV)
  }

  def treeReduce(f: (T, T) => T, depth: Int = 2): T = {
    val seqOp: (Option[T], T) => Option[T] = {
      case (Some(l), r) => Some(f(l, r))
      case (None, r) => Some(r)
    }

    val combOp: (Option[T], Option[T]) => Option[T] = {
      case (Some(l), Some(r)) => Some(f(l, r))
      case (l: Some[_], None) => l
      case (None, r: Some[_]) => r
      case (None, None) => None
    }

    treeAggregate(() => Option.empty, seqOp, combOp, depth)
      .getOrElse(throw new RuntimeException("nothing in the RDD!"))
  }

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

  def cmapPartitionsWithIndex[U: ClassTag](
    f: (Int, C, Iterator[T]) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] = new ContextRDD(
    rdd.mapPartitionsWithIndex(
      (i, part) => inCtx(ctx => f(i, ctx, part.flatMap(_(ctx)))),
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

  // FIXME: not safe because same context goes to both but will
  // clear on every call to f's return value
  def czipPartitions[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (C, Iterator[T], Iterator[U]) => Iterator[V]
  ): ContextRDD[C, V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)(
      (l, r) => inCtx(ctx => f(ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))),
    mkc)

//  def safeCZipPartitions[U: ClassTag, V: ClassTag](
//    that: ContextRDD[C, U],
//    preservesPartitioning: Boolean = false
//  )(f: (C, Iterator[T], Iterator[U]) => Iterator[V]
//  ): ContextRDD[C, V] = new ContextRDD(
//    czipPartitionsAndContext(that, preservesPartitioning) { (ctx, leftProducer, rightProducer) =>
//      val leftCtx = mkc()
//      val rightCtx = mkc()
//      val l = new SetupIterator(leftProducer.flatMap(_ (leftCtx)), () => leftCtx.reset())
//      val r = new SetupIterator(rightProducer.flatMap(_ (rightCtx)), () => rightCtx.reset())
//      f(ctx, l, r)
//    },
//    mkc)

  def czipPartitionsAndContext[U: ClassTag, V: ClassTag](
    that: ContextRDD[C, U],
    preservesPartitioning: Boolean = false
  )(f: (C, Iterator[C => Iterator[T]], Iterator[C => Iterator[U]]) => Iterator[V]
  ): ContextRDD[C, V] = new ContextRDD(
    rdd.zipPartitions(that.rdd, preservesPartitioning)(
      (l, r) => inCtx(ctx => f(ctx, l, r))),
    mkc)

  def cmapPartitionsAndContext[U: ClassTag](
    f: (C, (Iterator[C => Iterator[T]])) => Iterator[U],
    preservesPartitioning: Boolean = false
  ): ContextRDD[C, U] =
    onRDD(_.mapPartitions(
      part => inCtx(ctx => f(ctx, part)),
      preservesPartitioning))

  def subsetPartitions(keptPartitionIndices: Array[Int]): ContextRDD[C, T] =
    onRDD(_.subsetPartitions(keptPartitionIndices))

  def reorderPartitions(oldIndices: Array[Int]): ContextRDD[C, T] =
    onRDD(_.reorderPartitions(oldIndices))

  def adjustPartitions(
    adjustments: IndexedSeq[Array[Adjustment[T]]]
  ): ContextRDD[C, T] = {
    def contextIgnorantPartitionFunction(
      f: Iterator[T] => Iterator[T]
    ): Iterator[C => Iterator[T]] => Iterator[C => Iterator[T]] =
      it => inCtx(ctx => f(it.flatMap(useCtx => useCtx(ctx))))
    def contextIgnorantAdjustment(a: Adjustment[T]): Adjustment[C => Iterator[T]] =
      Adjustment(a.index, contextIgnorantPartitionFunction(a.f))
    val contextIgnorantAdjustments =
      adjustments.map(as => as.map(a => contextIgnorantAdjustment(a)))
    onRDD(rdd => new AdjustedPartitionsRDD(rdd, contextIgnorantAdjustments))
  }

  def coalesce(numPartitions: Int, shuffle: Boolean = false): ContextRDD[C, T] =
    // NB: the run marks the end of a context lifetime, the next one starts
    // after the shuffle
    if (shuffle)
      ContextRDD.weaken(run.coalesce(numPartitions, shuffle), mkc)
    else
      onRDD(_.coalesce(numPartitions, shuffle))

  def sample(
    withReplacement: Boolean,
    fraction: Double,
    seed: Long
  ): ContextRDD[C, T] = {
    require(fraction >= 0.0 && fraction <= 1.0)
    new ContextRDD(
      (if (withReplacement)
        new ContextSampledRDD(rdd, new PoissonSampler(fraction), seed)
      else
        new ContextSampledRDD(rdd, new BernoulliSampler(fraction), seed)),
      mkc)
  }

  def partitionSizes: Array[Long] =
    sparkContext.runJob(rdd, { (it: Iterator[ElementType]) =>
      using(mkc()) { c =>
        val it2 = it.flatMap(_(c))
        var count = 0L
        while (it2.hasNext) {
          it2.next()
          c.reset()
          count += 1
        }
        count
      }
    })

  def sparkContext: SparkContext = rdd.sparkContext

  def getNumPartitions: Int = rdd.getNumPartitions

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

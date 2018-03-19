package is.hail.sparkextras

import is.hail.io._
import is.hail.expr.types._
import is.hail.utils._

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.storage._
import scala.reflect.ClassTag

object ContextRDD {
  def empty[C <: AutoCloseable, T: ClassTag](sc: SparkContext): ContextRDD[C, T] =
    new ContextRDD(sc.emptyRDD[C => Iterator[T]])

  def union[C <: AutoCloseable, T: ClassTag]
    (sc: SparkContext, xs: Seq[ContextRDD[C, T]])
      : ContextRDD[C, T] =
    new ContextRDD(sc.union(xs.map(_.rdd)))

  // def weaken[C, T: ClassTag](rdd: RDD[T]): ContextRDD[C, T] =
  //   new ContextRDD(rdd.mapPartitions(it => Iterator.single((ctx: C) => it)))

  // this one weird trick permits the caller to specify C without T
  sealed trait Weaken[C <: AutoCloseable] {
    def apply[T: ClassTag](rdd: RDD[T]): ContextRDD[C, T] =
      new ContextRDD(rdd.mapPartitions(it => Iterator.single((ctx: C) => it)))
  }
  private[this] object weakenInstance extends Weaken[Nothing]
  def weaken[C <: AutoCloseable] = weakenInstance.asInstanceOf[Weaken[C]]

  type ElementType[C, T] = C => Iterator[T]
}

// FIXME: Subclassing seems super dangerous, did I actually implement the
// interace right?!?! Also: I want `map` to work on `T` not on the literal
// elements of the RDD (i.e. `C => Iterator[T]`).
class ContextRDD[C <: AutoCloseable, T: ClassTag](val rdd: RDD[C => Iterator[T]])
    extends Serializable {
  // FIXME: is this the right choice? Should I broadcast c? Should there be a
  // new one per useCtx? Probably need a method that constructs a context
  // FIXME: I probably need to reset the region per record??
  def run(c: () => C): RDD[T] =
    rdd.mapPartitions { part => using(c()) { cc => part.flatMap(_(cc)).toArray.iterator } }

  def collect(c: () => C): Array[T] = run(c).collect()

  private[this] def inCtx[U: ClassTag](f: C => Iterator[U])
      : Iterator[C => Iterator[U]] =
    Iterator.single { (ctx: C) => f(ctx) }

  private[this] def withoutContext[U: ClassTag](f: Iterator[T] => Iterator[U])
      : ContextRDD[C, U] =
    new ContextRDD(rdd.map(_.andThen(f)))

  def map[U: ClassTag](f: T => U): ContextRDD[C, U] =
    withoutContext(_.map(f))

  def filter(f: T => Boolean): ContextRDD[C, T] =
    withoutContext(_.filter(f))

  def flatMap[U: ClassTag](f: T => TraversableOnce[U]): ContextRDD[C, U] =
    withoutContext(_.flatMap(f))

  def mapPartitions[U: ClassTag]
    (f: Iterator[T] => Iterator[U], preservesPartitioning: Boolean = false)
      : ContextRDD[C, U] =
    cmapPartitions((_, part) => f(part), preservesPartitioning)

  def mapPartitionsWithIndex[U: ClassTag](f: (Int, Iterator[T]) => Iterator[U])
      : ContextRDD[C, U] =
    cmapPartitionsWithIndex((i, _, part) => f(i, part))

  // context is shared
  def zipPartitions[U: ClassTag, V: ClassTag]
    (that: ContextRDD[C, U], preservesPartitioning: Boolean = false)
    (f: (Iterator[T], Iterator[U]) => Iterator[V])
      : ContextRDD[C, V] =
    czipPartitions[U, V](that, preservesPartitioning)((_, l, r) => f(l, r))

  // FIXME: should this even exist? Where am I cleaning up the context?
  def treeAggregate[U: ClassTag](c: () => C)(zero: U)
    (seqOp: (U, T) => U,combOp: (U, U) => U, depth: Int = 2): U =
    rdd.treeAggregate(zero)(
      // FIXME: shouldn't this be per-partition?? Should I ensure there's one
      // iterator per partition?
      { (u, useCtx) => useCtx(c()).foldLeft(u)(seqOp)},
      combOp,
      depth)

  // FIXME: is this just treeAggregate with depth = 1?
  def aggregate[U: ClassTag](c: () => C)(zero: U)
    (seqOp: (U, T) => U,combOp: (U, U) => U): U =
    rdd.aggregate(zero)(
      // FIXME: shouldn't this be per-partition?? Should I ensure there's one
      // iterator per partition?
      { (u, useCtx) => useCtx(c()).foldLeft(u)(seqOp)},
      combOp)

  def aggregateByKey[U: ClassTag, K: ClassTag, V: ClassTag]
    (zero: U)(seqOp: (U, V) => U,combOp: (U, U) => U)
    (implicit ev: T =:= (K, V))
      : ContextRDD[C, (K, U)] =
    // FIXME: how do I do this
    ???

  def summarizePartitions[U: ClassTag](c: () => C)(f: Iterator[T] => U): Array[U] =
    run(c).mapPartitions(f.andThen(Iterator.single)).collect()

  def find(f: T => Boolean): Option[T] = filter(f).take(1) match {
    case Array(elem) => Some(elem)
    case _ => None
  }

  def take(i: Int): Array[T] =
    ??? // FIXME: I shouldn't need a context to do this

  private[this] def withContext[U: ClassTag](f: (C, Iterator[T]) => Iterator[U])
      : ContextRDD[C, U] =
    new ContextRDD(rdd.map(useCtx => ctx => f(ctx, useCtx(ctx))))

  def cmap[U: ClassTag](f: (C, T) => U): ContextRDD[C, U] =
    withContext((c, it) => it.map(f(c,_)))

  def cfilter(f: (C, T) => Boolean): ContextRDD[C, T] =
    withContext((c, it) => it.filter(f(c,_)))

  def cflatMap[U: ClassTag](f: (C, T) => TraversableOnce[U]): ContextRDD[C, U] =
    withContext((c, it) => it.flatMap(f(c,_)))

  def cmapPartitions[U: ClassTag]
    (f: (C, Iterator[T]) => Iterator[U], preservesPartitioning: Boolean = false)
      : ContextRDD[C, U] =
    new ContextRDD(rdd.mapPartitions(
      part => inCtx(ctx => f(ctx, part.flatMap(_(ctx)))),
      preservesPartitioning))

  def cmapPartitionsWithIndex[U: ClassTag](f: (Int, C, Iterator[T]) => Iterator[U])
      : ContextRDD[C, U] =
    new ContextRDD(rdd.mapPartitionsWithIndex(
      (i, part) => inCtx(ctx => f(i, ctx, part.flatMap(_(ctx))))))

  def czipPartitions[U: ClassTag, V: ClassTag]
    (that: ContextRDD[C, U], preservesPartitioning: Boolean = false)
    (f: (C, Iterator[T], Iterator[U]) => Iterator[V])
      : ContextRDD[C, V] =
    new ContextRDD(rdd.zipPartitions(that.rdd, preservesPartitioning)(
      (l, r) => inCtx(ctx => f(ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))))

  def sample(c: () => C)(withReplacement: Boolean, p: Double, seed: Long)
      : ContextRDD[C, T] =
    ContextRDD.weaken(run(c).sample(withReplacement, p, seed))

  // delagate to Spark

  // WTF SPARK? Two methods for the same thing?
  def sparkContext: SparkContext = rdd.sparkContext
  def context: SparkContext = rdd.context

  def getNumPartitions: Int = rdd.getNumPartitions

  def partitions: Array[Partition] = rdd.partitions

  def partitioner: Option[Partitioner] = rdd.partitioner

  def persist(level: StorageLevel): ContextRDD[C, T] =
    onRdd(_.persist)

  def unpersist(): ContextRDD[C, T] =
    onRdd(_.unpersist())

  def getStorageLevel: StorageLevel = rdd.getStorageLevel

  def coalesce(maxPartitions: Int, shuffle: Boolean): ContextRDD[C, T] =
    onRdd(_.coalesce(maxPartitions, shuffle))

  def preferredLocations(p: Partition): Seq[String] =
    rdd.preferredLocations(p)

  def subsetPartitions(keptPartitionIndices: Array[Int]): ContextRDD[C, T] =
    onRdd(_.subsetPartitions(keptPartitionIndices))

  def head(c: () => C)(n: Long): ContextRDD[C, T] =
    // FIXME: do I really need a context to do head? This doesn't seem
    // necessary. I think I just need to pull the thread on RichRDD.head
    ContextRDD.weaken(run(c).head(n))

  def reorderPartitions(oldIndices: Array[Int]): ContextRDD[C, T] =
    onRdd(_.reorderPartitions(oldIndices))

  def adjustPartitions(adjustments: IndexedSeq[Array[Adjustment[T]]])
      : ContextRDD[C, T] = {
    def contextIgnorantPartitionFunction(f: Iterator[T] => Iterator[T])
        : Iterator[C => Iterator[T]] => Iterator[C => Iterator[T]] =
      it => inCtx(ctx => f(it.flatMap(useCtx => useCtx(ctx))))
    def contextIgnorantAdjustment(a: Adjustment[T]): Adjustment[C => Iterator[T]] =
      new Adjustment(a.index, contextIgnorantPartitionFunction(a.f))
    val contextIgnorantAdjustments =
      adjustments.map(as => as.map(a => contextIgnorantAdjustment(a)))
    onRdd(rdd => new AdjustedPartitionsRDD(rdd, contextIgnorantAdjustments))
  }

  def partitionBy[K: ClassTag, V: ClassTag](partitioner: Partitioner)
    (implicit ev: (K, V) =:= T): ContextRDD[C, T] =
    // FIXME: this actually inserts a map step that shouldn't be necessary
    // probably need PairContextRDDFunctions
    /// map(ev(_)).onRdd(_.partitionBy(partitioner))
    // FIXME: Actually, wtf, how do I do this? I need a new RDD I think
    ???

  def values[K: ClassTag, V: ClassTag](implicit ev: (K, V) =:= T)
      : ContextRDD[C, V] =
    // FIXME scala is actually super dumb and doesn't let me go from T to (K, V)
    // using ev /shrug
    ???
    // map(???.apply(_)._2)

  def iterator(partition: Partition, context: TaskContext): Iterator[C => Iterator[T]] =
    rdd.iterator(partition, context)

  // FIXME: this seems like a dangerous method, I should suck all functionality
  // into my interface and this should be private. I think only use is
  // BlockedRDD in OrderedRVD; should I just implement that here instead?
  def onRdd(f: RDD[C => Iterator[T]] => RDD[C => Iterator[T]]): ContextRDD[C, T] =
    new ContextRDD(f(rdd))

  // FIXME: I should define cache and use it smartly rather than run'ing. go
  // look for all run's followed by cache and fix them
}

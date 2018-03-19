package is.hail.sparkextras

import is.hail.io._
import is.hail.expr.types._
import is.hail.utils._

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.storage._
import scala.reflect.ClassTag

object ContextRDD {
  def empty[C <: AutoCloseable, T: ClassTag](
    sc: SparkContext,
    mkc: () => C
  ): ContextRDD[C, T] =
    new ContextRDD(sc.emptyRDD[C => Iterator[T]], mkc)

  def union[C <: AutoCloseable, T: ClassTag](
    sc: SparkContext,
    xs: Seq[ContextRDD[C, T]]
  ): ContextRDD[C, T] = {
    assert(xs.length > 1)
    val mkc = xs(0).mkc
    assert(xs.tail.forall(_ == mkc))
    union(sc, xs, mkc)
  }

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
    def apply[T: ClassTag](rdd: RDD[T], mkc: () => C): ContextRDD[C, T] =
      new ContextRDD(
        rdd.mapPartitions(it => Iterator.single((ctx: C) => it)),
        mkc)
  }
  private[this] object weakenInstance extends Weaken[Nothing]
  def weaken[C <: AutoCloseable] = weakenInstance.asInstanceOf[Weaken[C]]

  type ElementType[C, T] = C => Iterator[T]
}

class ContextRDD[C <: AutoCloseable, T: ClassTag](
  val rdd: RDD[C => Iterator[T]],
  val mkc: () => C
) extends Serializable {
  def run: RDD[T] =
    rdd.mapPartitions { part => using(mkc()) { cc => part.flatMap(_(cc)).toArray.iterator } }

  private[this] def inCtx[U: ClassTag](f: C => Iterator[U])
      : Iterator[C => Iterator[U]] =
    Iterator.single { (ctx: C) => f(ctx) }

  private[this] def withoutContext[U: ClassTag](f: Iterator[T] => Iterator[U])
      : ContextRDD[C, U] =
    new ContextRDD(rdd.map(_.andThen(f)), mkc)

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

  def zipPartitions[U: ClassTag, V: ClassTag]
    (that: ContextRDD[C, U], preservesPartitioning: Boolean = false)
    (f: (Iterator[T], Iterator[U]) => Iterator[V])
      : ContextRDD[C, V] =
    czipPartitions[U, V](that, preservesPartitioning)((_, l, r) => f(l, r))

  private[this] def withContext[U: ClassTag](f: (C, Iterator[T]) => Iterator[U])
      : ContextRDD[C, U] =
    new ContextRDD(rdd.map(useCtx => ctx => f(ctx, useCtx(ctx))), mkc)

  def cmap[U: ClassTag](f: (C, T) => U): ContextRDD[C, U] =
    withContext((c, it) => it.map(f(c,_)))

  def cfilter(f: (C, T) => Boolean): ContextRDD[C, T] =
    withContext((c, it) => it.filter(f(c,_)))

  def cflatMap[U: ClassTag](f: (C, T) => TraversableOnce[U]): ContextRDD[C, U] =
    withContext((c, it) => it.flatMap(f(c,_)))

  def cmapPartitions[U: ClassTag]
    (f: (C, Iterator[T]) => Iterator[U], preservesPartitioning: Boolean = false)
      : ContextRDD[C, U] =
    new ContextRDD(
      rdd.mapPartitions(
        part => inCtx(ctx => f(ctx, part.flatMap(_(ctx)))),
        preservesPartitioning),
      mkc)

  def cmapPartitionsWithIndex[U: ClassTag](f: (Int, C, Iterator[T]) => Iterator[U])
      : ContextRDD[C, U] =
    new ContextRDD(
      rdd.mapPartitionsWithIndex(
        (i, part) => inCtx(ctx => f(i, ctx, part.flatMap(_(ctx))))),
      mkc)

  def czipPartitions[U: ClassTag, V: ClassTag]
    (that: ContextRDD[C, U], preservesPartitioning: Boolean = false)
    (f: (C, Iterator[T], Iterator[U]) => Iterator[V])
      : ContextRDD[C, V] =
    new ContextRDD(
      rdd.zipPartitions(that.rdd, preservesPartitioning)(
        (l, r) => inCtx(ctx => f(ctx, l.flatMap(_(ctx)), r.flatMap(_(ctx))))),
      mkc)

  def subsetPartitions(keptPartitionIndices: Array[Int]): ContextRDD[C, T] =
    onRDD(_.subsetPartitions(keptPartitionIndices))

  def reorderPartitions(oldIndices: Array[Int]): ContextRDD[C, T] =
    onRDD(_.reorderPartitions(oldIndices))

  def adjustPartitions(adjustments: IndexedSeq[Array[Adjustment[T]]])
      : ContextRDD[C, T] = {
    def contextIgnorantPartitionFunction(f: Iterator[T] => Iterator[T])
        : Iterator[C => Iterator[T]] => Iterator[C => Iterator[T]] =
      it => inCtx(ctx => f(it.flatMap(useCtx => useCtx(ctx))))
    def contextIgnorantAdjustment(a: Adjustment[T]): Adjustment[C => Iterator[T]] =
      new Adjustment(a.index, contextIgnorantPartitionFunction(a.f))
    val contextIgnorantAdjustments =
      adjustments.map(as => as.map(a => contextIgnorantAdjustment(a)))
    onRDD(rdd => new AdjustedPartitionsRDD(rdd, contextIgnorantAdjustments))
  }

  private[this] def onRDD(f: RDD[C => Iterator[T]] => RDD[C => Iterator[T]]): ContextRDD[C, T] =
    new ContextRDD(f(rdd), mkc)
}

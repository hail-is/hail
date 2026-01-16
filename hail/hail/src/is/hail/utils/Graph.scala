package is.hail.utils

import is.hail.annotations.{Region, RegionValueBuilder, UnsafeIndexedSeq}
import is.hail.asm4s._
import is.hail.backend.{HailStateManager, HailTaskContext}
import is.hail.collection.BinaryHeap
import is.hail.io.fs.FS
import is.hail.types.physical.PTuple
import is.hail.variant.ReferenceGenome

import scala.collection.compat._
import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.sql.Row

object Graph {
  private type MultiMap[A, B] = mutable.Map[A, mutable.Set[B]]

  def mkGraph[T](edges: (T, T)*): MultiMap[T, T] =
    mkGraph(edges.toArray)

  def mkGraph[T](edges: Array[(T, T)]): MultiMap[T, T] = {
    val m: MultiMap[T, T] = mutable.HashMap.empty
    var i = 0
    while (i < edges.length) {
      m.getOrElseUpdate(edges(i)._1, mutable.Set.empty) += edges(i)._2
      m.getOrElseUpdate(edges(i)._2, mutable.Set.empty) += edges(i)._1
      i += 1
    }
    m
  }

  def mkGraph[T](edges: IterableOnce[(T, T)]): MultiMap[T, T] = {
    val m: MultiMap[T, T] = mutable.HashMap.empty.withDefault(_ => mutable.Set.empty)
    edges.iterator.foreach { case (i, j) =>
      m.getOrElseUpdate(i, mutable.Set.empty) += j
      m.getOrElseUpdate(j, mutable.Set.empty) += i
    }
    m
  }

  def maximalIndependentSet(edges: UnsafeIndexedSeq): IndexedSeq[Any] =
    maximalIndependentSet(mkGraph(edges.view.map { case Row(i, j) => i -> j }))

  def maximalIndependentSet(
    rgs: Map[String, ReferenceGenome],
    edges: UnsafeIndexedSeq,
    hcl: HailClassLoader,
    fs: FS,
    htc: HailTaskContext,
    outerRegion: Region,
    wrappedNodeType: PTuple,
    resultType: PTuple,
    tieBreaker: (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction3RegionLongLongLong,
  ): IndexedSeq[Any] = {
    val nodeType = wrappedNodeType.types.head.virtualType
    val region = outerRegion.getPool().getRegion()
    val tieBreakerF = tieBreaker(hcl, fs, htc, region)
    val rvb = new RegionValueBuilder(HailStateManager(rgs))
    val tbf = (l: Any, r: Any) => {
      region.clear()
      rvb.set(region)

      rvb.start(wrappedNodeType)
      rvb.startTuple()
      rvb.addAnnotation(nodeType, l)
      rvb.endTuple()
      val lOffset = rvb.end()

      rvb.start(wrappedNodeType)
      rvb.startTuple()
      rvb.addAnnotation(nodeType, r)
      rvb.endTuple()
      val rOffset = rvb.end()

      val resultOffset = tieBreakerF(region, lOffset, rOffset)
      if (resultType.isFieldMissing(resultOffset, 0)) {
        throw new RuntimeException(
          s"a comparison returned a missing value when " +
            s"l=${Region.pretty(wrappedNodeType, lOffset)} and r=${Region.pretty(wrappedNodeType, rOffset)}"
        )
      } else {
        Region.loadDouble(resultType.loadField(resultOffset, 0))
      }
    }

    maximalIndependentSet(mkGraph(edges.map { case Row(i, j) => i -> j }), Some(tbf))
  }

  def maximalIndependentSet[T: ClassTag](edges: Array[(T, T)]): IndexedSeq[T] =
    maximalIndependentSet(mkGraph(edges))

  def maximalIndependentSet[T: ClassTag](edges: Array[(T, T)], tieBreaker: (T, T) => Double)
    : IndexedSeq[T] =
    maximalIndependentSet(mkGraph(edges), Some(tieBreaker))

  def maximalIndependentSet[T: ClassTag](
    g: MultiMap[T, T],
    maybeTieBreaker: Option[(T, T) => Double] = None,
  ): IndexedSeq[T] = {
    val verticesByDegree = new BinaryHeap[T](maybeTieBreaker = maybeTieBreaker.orNull)

    g.foreach { case (v, neighbors) =>
      verticesByDegree.insert(v, neighbors.size.toLong)
    }

    while (verticesByDegree.nonEmpty && verticesByDegree.maxPriority() > 0) {
      val current = verticesByDegree.extractMax()
      for {
        x <- g(current)
        if x != current
      } {
        g(x) -= current
        verticesByDegree.decreasePriority(x, _ - 1)
      }
    }

    verticesByDegree.toArray
  }
}

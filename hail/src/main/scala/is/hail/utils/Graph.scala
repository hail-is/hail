package is.hail.utils

import is.hail.annotations.{Region, RegionValueBuilder, UnsafeIndexedSeq}
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailStateManager, HailTaskContext}
import is.hail.expr.ir.{
  Compile, Interpret, IR, IRParser, IRParserEnvironment, Literal, MakeTuple, SingleCodeEmitParamType,
}
import is.hail.io.fs.FS
import is.hail.types.physical.{stypes, PCanonicalTuple, PTuple, PType}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.sql.Row

object Graph {
  def mkGraph[T](edges: (T, T)*): mutable.MultiMap[T, T] =
    mkGraph(edges.toArray)

  def mkGraph[T](edges: Array[(T, T)]): mutable.MultiMap[T, T] = {
    val m = new mutable.HashMap[T, mutable.Set[T]]() with mutable.MultiMap[T, T]
    var i = 0
    while (i < edges.length) {
      m.addBinding(edges(i)._1, edges(i)._2)
      m.addBinding(edges(i)._2, edges(i)._1)
      i += 1
    }
    m
  }

  def mkGraph[T](edges: TraversableOnce[(T, T)]): mutable.MultiMap[T, T] = {
    val m = new mutable.HashMap[T, mutable.Set[T]]() with mutable.MultiMap[T, T]
    edges.foreach { case (i, j) =>
      m.addBinding(i, j)
      m.addBinding(j, i)
    }
    m
  }

  def maximalIndependentSet(edges: UnsafeIndexedSeq): IndexedSeq[Any] =
    maximalIndependentSet(mkGraph(edges.map { case Row(i, j) => i -> j }))

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
    g: mutable.MultiMap[T, T],
    maybeTieBreaker: Option[(T, T) => Double] = None,
  ): IndexedSeq[T] = {
    val verticesByDegree = new BinaryHeap[T](maybeTieBreaker = maybeTieBreaker.orNull)

    g.foreach { case (v, neighbors) =>
      verticesByDegree.insert(v, neighbors.size)
    }

    while (verticesByDegree.nonEmpty && verticesByDegree.maxPriority() > 0) {
      val current = verticesByDegree.extractMax()
      val neighbors = g(current) - current
      neighbors.foreach { x =>
        g(x) -= current
        verticesByDegree.decreasePriority(x, _ - 1)
      }
    }

    verticesByDegree.toArray
  }
}

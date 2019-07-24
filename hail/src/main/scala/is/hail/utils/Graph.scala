package is.hail.utils

import is.hail.annotations.{Region, RegionValueBuilder, SafeRow}
import is.hail.expr.ir.{Compile, IR, IRParser, IRParserEnvironment, Interpret, Literal, MakeTuple}
import is.hail.expr.types.physical.{PBaseStruct, PSet, PTuple, PType}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, IRParserEnvironment, Interpret, Literal, MakeTuple}
import is.hail.expr.types.physical.PBaseStruct
import is.hail.expr.types.virtual._
import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.reflect.ClassTag

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

  def pyMaximalIndependentSet(edgesIR: IR, nodeTypeStr: String, tieBreaker: Option[String]): IR = {
    val nodeType = IRParser.parseType(nodeTypeStr)
    
    val edges = ExecuteContext.scoped { ctx => Interpret[IndexedSeq[Row]](ctx, edgesIR).toArray }

    val resultType = TSet(nodeType)
    val result = maximalIndependentSet(edges, nodeType, tieBreaker)
    Literal(resultType, result)
  }

  def maximalIndependentSet(edges: Array[Row], nodeType: Type, tieBreaker: Option[String]): Set[Any] = {
    val edges2 = edges.map { r =>
      val Row(x, y) = r
      (x, y)
    }

    if (edges2.length > 400000)
      warn(s"over 400,000 edges are in the graph; maximal_independent_set may run out of memory")

    val wrappedNodeType = PTuple(PType.canonical(nodeType))
    val refMap = Map("l" -> wrappedNodeType.virtualType, "r" -> wrappedNodeType.virtualType)

    val tieBreakerF = tieBreaker.map { e =>
      val ir = IRParser.parse_value_ir(e, IRParserEnvironment(refMap))
      val (t, f) = Compile[Long, Long, Long]("l", wrappedNodeType, "r", wrappedNodeType, MakeTuple.ordered(FastSeq(ir)))
      assert(t.virtualType.isOfType(TTuple(TInt64())))

      (l: Any, r: Any) => {
        Region.scoped { region =>
          val rvb = new RegionValueBuilder()
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

          val resultOffset = f(0, region)(region, lOffset, false, rOffset, false)
          SafeRow(t.asInstanceOf[PBaseStruct], region, resultOffset).get(0).asInstanceOf[Long]
        }
      }
    }

    maximalIndependentSet(mkGraph(edges2), tieBreakerF).toSet
  }

  def maximalIndependentSet[T: ClassTag](edges: Array[(T, T)]): Array[T] = {
    maximalIndependentSet(mkGraph(edges))
  }

  def maximalIndependentSet[T: ClassTag](edges: Array[(T, T)], tieBreaker: (T, T) => Long): Array[T] = {
    maximalIndependentSet(mkGraph(edges), Some(tieBreaker))
  }

  def maximalIndependentSet[T: ClassTag](g: mutable.MultiMap[T, T], maybeTieBreaker: Option[(T, T) => Long] = None): Array[T] = {
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

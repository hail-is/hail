package is.hail.utils

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

  def maximalIndependentSet[T: ClassTag](edges: Array[(T, T)]): Array[T] = {
    maximalIndependentSet(mkGraph(edges))
  }

  def maximalIndependentSet[T: ClassTag](g: mutable.MultiMap[T, T]): Array[T] = {
    val verticesByDegree = new BinaryHeap[T]()

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

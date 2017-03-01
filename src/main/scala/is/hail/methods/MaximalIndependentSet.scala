package is.hail.methods

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._
import is.hail.utils._

import scala.reflect.ClassTag

object MaximalIndependentSet {

  def apply[VD: ClassTag, ED: ClassTag](g: Graph[VD, ED], undirected: Boolean = false): Set[Long] = {

    type Message = (Int, VertexId)
    val pairOrd = implicitly[Ordering[Message]]
    import pairOrd.mkOrderingOps

    val initialMsg = (-1, -1L)

    def receiveMessage(vertexId: VertexId, value: Message, message: Message): Message = {
      value max message
    }

    def sendMsg(triplet: EdgeTriplet[Message, ED]): Iterator[(VertexId, Message)] = {
      if (triplet.srcAttr > triplet.dstAttr)
        Iterator((triplet.dstId, triplet.srcAttr))
      else if (triplet.srcAttr < triplet.dstAttr && undirected)
        Iterator((triplet.srcId, triplet.dstAttr))
      else
        Iterator.empty
    }

    def mergeMsg(x: Message, y: Message) =
      x max y

    def updateVertexDegrees(toBeComputed: Graph[_, ED]): Graph[Message, ED] = {
      Graph(toBeComputed.vertices.leftZipJoin(toBeComputed.degrees) { (v, _, degree) => (degree.getOrElse(0), v) }, toBeComputed.edges)
    }

    var graph1: Graph[Message, ED] = updateVertexDegrees(g)
    var graph2: Graph[Message, ED] = null
    val edgeDirection = if (undirected) EdgeDirection.Either else EdgeDirection.Out

    while(graph1.numEdges > 0) {
      if (graph2 != null) {
        graph2.unpersist()
      }

      val pregelGraph = graph1.pregel(initialMsg, Int.MaxValue, edgeDirection)(receiveMessage, sendMsg, mergeMsg)
      val idSet = pregelGraph.vertices
        .filter(tuple => tuple match {case (id, value) => value match  {case (maxDegrees, maxID) => maxID == id && maxDegrees != 0}})
        .map(_._1).collect().toSet
      var newGraph = graph1.subgraph(_ => true, (id, value) => !idSet.contains(id))
      newGraph = updateVertexDegrees(newGraph)
      graph2 = graph1
      graph1 = newGraph
      graph1 = graph1.persist()
    }
    if (graph2 != null) {
      graph2.unpersist()
    }
    graph1.vertices.keys.collect().toSet
  }


  def ofIBDMatrix(inputRDD: RDD[((Int, Int), Double)], thresh: Double, vertexIDs: Seq[Int]): Set[Long] = {
    val sc = inputRDD.sparkContext

    val filteredRDD = inputRDD.filter(_._2 >= thresh)

    val edges: RDD[Edge[Double]] = filteredRDD.map{case((v1, v2), weight) => Edge(v1, v2, weight)}

    val vertices: RDD[(VertexId, Null)] = sc.parallelize(vertexIDs).map(id => (id, null))

    val graph: Graph[Null, Double] = Graph(vertices, edges)

    apply(graph, undirected = true)
  }
}

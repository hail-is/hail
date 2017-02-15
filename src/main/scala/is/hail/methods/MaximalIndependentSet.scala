package is.hail.methods

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

import scala.reflect.ClassTag

object MaximalIndependentSet {

  def apply[VD: ClassTag, ED: ClassTag](g: Graph[VD, ED], undirected: Boolean = false): Set[Long] = {
    // Initially set each vertex to its own degree.
    // Start a pregel run, everyone passing a (degree, ID) pair.
    //  -On message reception, if current degree is greater than received degree, update status to to reflect this, alert neighbors
    //  -If current degree < received degree, status does not change, don't bother sending alert.

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

    var graph = updateVertexDegrees(g)
    val edgeDirection = if (undirected) EdgeDirection.Either else EdgeDirection.Out

    while(graph.numEdges > 0) {
      graph = graph.pregel(initialMsg, Int.MaxValue, edgeDirection)(receiveMessage, sendMsg, mergeMsg)
      graph = graph.subgraph(_ => true, (id, value) => value match { case (maxDegrees, maxID) => maxID != id || maxDegrees == 0})
      graph = updateVertexDegrees(graph)
    }
    graph.vertices.keys.collect().toSet
  }


  def ofIBDMatrix(sc: SparkContext, inputRDD: RDD[((String, String), Double)], thresh: Double): Set[String] = {
    //Filter RDD to remove edges above threshold
    val filteredRDD = inputRDD.filter(_._2 <= thresh)

    //Throw away weights
    val vertexPairs = inputRDD.keys

    //Collect all vertices.
    val allVertices = vertexPairs.flatMap[String]{case (v1, v2) => List(v1, v2)}.distinct()

    val numberedVertices: RDD[(String, VertexId)] = allVertices.zipWithIndex()

    val verticesToNumbers = numberedVertices.collectAsMap()
    val numbersToVertices = numberedVertices.map{case (name, id) => (id, name)}.collectAsMap()


    val edges: RDD[Edge[Double]] = filteredRDD.map{case((v1, v2), weight) => Edge(verticesToNumbers(v1), verticesToNumbers(v2), weight)}

    val vertices: VertexRDD[String] = VertexRDD[String](numberedVertices.map{ case (id, index) => (index, id)})

    val stringGraph: Graph[String, Double] = Graph(vertices, edges)

    val mis = apply(stringGraph, true)

    mis.map(numbersToVertices(_))
  }
}

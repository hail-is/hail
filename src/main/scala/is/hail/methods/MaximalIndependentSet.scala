package is.hail.methods

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

import scala.reflect.ClassTag

object MaximalIndependentSet {

  def apply[VD: ClassTag, ED: ClassTag](g: Graph[VD, ED]): Set[Long] = {
    // Initially set each vertex to its own degree.
    // Start a pregel run, everyone passing a (ID, degree) pair.
    //  -On message reception, if current degree is greater than received degree, update status to to reflect this, alert everyone
    //  -If current degree <= received degree, status does not change, don't bother sending alert.

    //Every vertex needs to know
    // -Whether it changed recently (some facet of old state must be stored)
    // -What the highest degree message received has been
    // -What vertex is in that highest degree message

    var graph = Graph[((VertexId, Int), VertexId), ED](g.collectNeighborIds(EdgeDirection.Either)
      .map{ case(v, neighbors) => (v, ((v, neighbors.size), -1L))}, g.edges: RDD[Edge[ED]])

    val initialMsg = (-1L, -1)

    //vertexId is ID of the receiver
    def vprog(vertexId: VertexId, value: ((VertexId, Int), VertexId), message: (VertexId, Int)): ((VertexId, Int), VertexId) = {
      val ((currentVertex, currentMaxDegree), oldVertex) = value
      val (receivedVertex, receivedMaxDegree) = message


      if (message == initialMsg) {
        value
      }
      else if (currentMaxDegree < receivedMaxDegree) {
        (message, currentVertex)
      }
      else {
        (value._1, currentVertex)
      }

    }

    def sendMsg(triplet: EdgeTriplet[((VertexId, Int), VertexId), ED]): Iterator[(VertexId, (VertexId, Int))] = {
      val ((srcMaxID, srcMaxDegrees), srcOldID) = triplet.srcAttr

      if (srcMaxID == srcOldID) {
        Iterator.empty
      }
      else {
        Iterator((triplet.dstId, (srcMaxID, srcMaxDegrees)))
      }
    }

    def mergeMsg(msg1: (VertexId, Int), msg2: (VertexId, Int)): (VertexId, Int) = {
      if (msg1._2 > msg2._2) msg1 else msg2
    }

    while(graph.numEdges > 0) {
      graph = graph.pregel(initialMsg, Int.MaxValue, EdgeDirection.Both)(vprog, sendMsg, mergeMsg)
      graph = graph.subgraph(_ => true, (id, value) => value match { case ((maxID, maxDegrees), oldID) => maxID != oldID || maxDegrees == 0})
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

    val mis = apply(stringGraph)

    mis.map(numbersToVertices(_))

  }
}

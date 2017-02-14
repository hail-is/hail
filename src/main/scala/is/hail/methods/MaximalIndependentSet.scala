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
      val sourceAttr = triplet.srcAttr

      if (sourceAttr._1._1 == sourceAttr._2) {
        Iterator.empty
      }
      else {
        Iterator((triplet.dstId, sourceAttr._1))
      }
    }

    def mergeMsg(msg1: (VertexId, Int), msg2: (VertexId, Int)): (VertexId, Int) = {
      if (msg1._2 > msg2._2) msg1 else msg2
    }

    var atLeastOneEdge = graph.numEdges > 0
    while(atLeastOneEdge) {
      graph = graph.pregel(initialMsg, Int.MaxValue, EdgeDirection.Both)(vprog, sendMsg, mergeMsg)
      graph = graph.subgraph(_ => true, (id, value) => value._1._1 != value._2 || value._1._2 == 0)

      if (graph.numEdges == 0) {
        atLeastOneEdge = false
      }
    }

    graph.vertices.map{ case(id, _) => id}.collect().toSet
  }


  def ofIBDMatrix(sc: SparkContext, inputRDD: RDD[((String, String), Double)], thresh: Double): Set[String] = {
    //Filter RDD to remove edges above threshold
    val filteredRDD = inputRDD.filter(_._2 <= thresh)

    //Throw away weights
    val vertexPairs = inputRDD.map(_._1)

    //Collect all vertices.
    val allVertices = vertexPairs.flatMap[String](vertexPair => List(vertexPair._1, vertexPair._2)).distinct()

    val numberedVertices: RDD[(String, VertexId)] = allVertices.zipWithIndex()

    val verticesToNumbers = numberedVertices.collectAsMap()
    val numbersToVertices = numberedVertices.map(pair => (pair._2, pair._1)).collectAsMap()


    val edges: RDD[Edge[Double]] = filteredRDD.map(tuple => Edge(verticesToNumbers(tuple._1._1), verticesToNumbers(tuple._1._2), tuple._2))

    val vertices: VertexRDD[String] = VertexRDD[String](numberedVertices.map{ case (id, index) => (index, id)})

    val stringGraph: Graph[String, Double] = Graph(vertices, edges)

    val misGraph = apply(stringGraph)

    misGraph.map(numbersToVertices(_))

  }
}

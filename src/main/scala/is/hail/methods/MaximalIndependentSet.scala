package is.hail.methods

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

object MaximalIndependentSet {

  def apply(sc: SparkContext, inputRDD: RDD[((String, String), Double)], thresh: Double, k: Int): Set[String] = {
    //Filter RDD to remove edges above threshold
    val filteredRDD = inputRDD.filter(_._2 <= thresh)

    //Throw away weights
    val vertexPairs = inputRDD.map(_._1)

    //Collect all vertices.
    val allVertices = vertexPairs.flatMap[String](vertexPair => List(vertexPair._1, vertexPair._2)).distinct()

    val numberedVertices: RDD[(String, VertexId)] = allVertices.zip(sc.parallelize(0L until allVertices.count()))

    val verticesToNumbers = numberedVertices.collectAsMap()
    val numbersToVertices = numberedVertices.map(pair => (pair._2, pair._1)).collectAsMap()


    val edges: RDD[Edge[Double]] = filteredRDD.map(tuple => Edge(verticesToNumbers(tuple._1._1), verticesToNumbers(tuple._1._2), tuple._2))

    val vertices: VertexRDD[String] = VertexRDD[String](numberedVertices.map(pair => (pair._2, pair._1)))

    val stringGraph = Graph(vertices, edges)

    var graph = Graph[((VertexId, Int), VertexId), Double](stringGraph.degrees.map(pair => (pair._1, (pair, -1L))), stringGraph.edges)

    println(numbersToVertices)

    //graph.cache()

    //Initially set each vertex to its own degree.
    //Start a pregel run, everyone passing a (ID, degree) pair.
    //  -On message reception, if current degree is greater than received degree, update status to to reflect this, alert everyone
    //  -If current degree <= received degree, status does not change, don't bother sending alert.

    //Every vertex needs to know
    // -Whether it changed recently (some facet of old state must be stored)
    // -What the highest degree message received has been
    // -What vertex is in that highest degree message

    val initialMsg = (-1L, -1)

    //vertexId is ID of the receiver
    def vprog(vertexId: VertexId, value: ((VertexId, Int), VertexId), message: (VertexId, Int)): ((VertexId, Int), VertexId) = {
      val ((currentVertex, currentMaxDegree), oldVertex) = value
      val (receivedVertex, receivedMaxDegree) = message

      println("The message " + vertexId + " receives is " + message)

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

    def sendMsg(triplet: EdgeTriplet[((VertexId, Int), VertexId), Double]): Iterator[(VertexId, (VertexId, Int))] = {
      val sourceAttr = triplet.srcAttr

      if (sourceAttr._1._1 == sourceAttr._2) {
        Iterator.empty
      }
      else {
        println("" + triplet.srcId + " is sending a message to " + triplet.dstId)
        Iterator((triplet.dstId, sourceAttr._1))
      }
    }

    def mergeMsg(msg1: (VertexId, Int), msg2: (VertexId, Int)): (VertexId, Int) = {
      if (msg1._2 > msg2._2) msg1 else msg2
    }

    var atLeastOneEdge = graph.numEdges > 0
    var iterations = 0

    while(atLeastOneEdge) {
      graph = graph.pregel(initialMsg, Int.MaxValue, EdgeDirection.Both)(vprog, sendMsg, mergeMsg)
      graph = graph.subgraph(_ => true, (id, value) => value._1._1 != value._2)

      if (graph.numEdges == 0) {
        atLeastOneEdge = false
      }
      iterations += 1
    }
    print(iterations)

    val x: Array[(VertexId, ((VertexId, Int), VertexId))] = graph.vertices.collect()
    val y: Array[String] = x.map(tuple => numbersToVertices(tuple._1))

    y.toSet
    //merge message is correct

    //graph.pregel()

    /*def mis(graph: Graph[_, Double]): Graph[_, Double] = {
      //Steps:
      //Perform CC, call recursively on separate graphs.
      //If there are no more CC, perform the take, subgraph operation.
      //On this new graph, call function again

      if(graph.numEdges == 0) {
        return graph
      }

      val grouped: VertexRDD[VertexId] = graph.connectedComponents().vertices

      if (grouped.count() > 1) {
        //Apply mis to all graphs, fuse them.
        grouped.map(mis(sc.parallelize(_._2))
      }
      else {
        val topK = graph.collectNeighborIds(EdgeDirection.Either).sortBy(-_._2.size).take(k)

        val filteredVertices: Seq[(VertexId, Array[VertexId])] = topK.toList match {
          case x :: xs => x :: xs.filter(_._2.contains(x))
          case Nil => Nil
        }

        val vertexIds = filteredVertices.map(_._1)

        mis(graph.subgraph(_ => true, (id, name) => !vertexIds.contains(id)))
      }
    }*/

    //mis(graph).mapVertices((id, _) => numbersToVertices(id)).vertices.map(_._2).collect().toSet
  }
}

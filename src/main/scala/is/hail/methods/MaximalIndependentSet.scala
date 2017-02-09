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

    //Collect all vertices, map em to numbers
    val vertexMapping = vertexPairs.flatMap[String](vertexPair => List(vertexPair._1, vertexPair._2))
                                                        .distinct()
                                                        .zipWithUniqueId().collectAsMap()
    println(vertexMapping)
    //Consider wrapping EdgeRDD
    val edges: RDD[Edge[Double]] = filteredRDD.map(tuple => Edge(vertexMapping(tuple._1._1), vertexMapping(tuple._1._2), tuple._2))

    val vertices: VertexRDD[String] = VertexRDD[String](sc.parallelize(vertexMapping.toSeq).map(pair => (pair._2, pair._1)))

    var graph = Graph(vertices, edges)


    graph.cache()
    while((() => graph.numEdges > 0)()) {
      val topK = graph.collectNeighborIds(EdgeDirection.Either).sortBy(-_._2.size).take(k)

      val filteredVertices: Seq[(VertexId, Array[VertexId])] = topK.toList match {
        case x :: xs => x :: xs.filter(_._2.contains(x))
        case Nil => Nil
      }

      val vertexIds = filteredVertices.map(_._1)

      graph = graph.subgraph(_ => true, (id, name) => !vertexIds.contains(id))
    }

    graph.vertices.map(_._2).collect().toSet
  }
}

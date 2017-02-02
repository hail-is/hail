package is.hail.methods

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

/**
  * Created by johnc on 1/30/17.
  */
object VertexCover {

  def apply(sc: SparkContext, inputRDD: RDD[((String, String), Double)], thresh: Double, k: Int): Set[String] = {

    //Filter RDD to remove edges below(?) threshold
    val filteredRDD = inputRDD.filter(_._2 >= thresh)

    //Throw away weights
    val vertexPairs = filteredRDD.map(_._1)

    //Collect all vertices, map em to numbers
    val vertexMapping = vertexPairs.flatMap[String](vertexPair => List(vertexPair._1, vertexPair._2))
                                                        .distinct()
                                                        .zipWithUniqueId().collectAsMap()

    val edges = filteredRDD.map(tuple => Edge(vertexMapping(tuple._1._1), vertexMapping(tuple._1._2), tuple._2))

    val vertices = sc.parallelize(vertexMapping.toSeq).map(pair => (pair._2, pair._1))

    val graph = Graph(vertices, edges)


    

    //Get top k vertices
    graph.degrees.sortBy(_._2).take(k)

  }
}

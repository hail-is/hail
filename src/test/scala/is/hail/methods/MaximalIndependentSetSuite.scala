package is.hail.methods

import is.hail.SparkSuite
import org.apache.spark.graphx.{Edge, Graph, VertexId, VertexRDD}
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/**
  * Created by johnc on 2/8/17.
  */
class MaximalIndependentSetSuite extends SparkSuite {

  @Test def emptySet() {
    assert(MaximalIndependentSet(Graph(sc.parallelize(Array[(VertexId, Int)]()),
      sc.parallelize(Array[Edge[Int]]()))) == Set())
  }

  @Test def graphWithoutEdges() {
    val vertices = sc.parallelize(Array[(VertexId, String)](
      (1, "A"), (2, "B"), (3, "C")
    ))

    val edges = sc.parallelize(Array[Edge[Int]]())

    assert(MaximalIndependentSet(Graph(vertices, edges)) == Set(1, 2, 3))
  }

  @Test def simpleGraph() {
    val vertices = sc.parallelize(Array[(VertexId, String)](
      (1, "A"), (2, "B"), (3, "C")
    ))

    val edges = sc.parallelize(Array[Edge[String]](
      Edge(1, 2, "AB"), Edge(2, 3, "BC")
    ))

    assert(MaximalIndependentSet(Graph(vertices, edges)) == Set(1, 3))
  }

  @Test def ofIBDMatrixTest1() {
    val input: RDD[((String, String), Double)] = sc.parallelize(Array(
      (("A", "B"), 0.3), (("B", "C"), 0.3), (("B", "D"), 0.3),
      (("D", "F"), 0.0), (("F", "E"), 0.0), (("G", "F"), 0.0),
      (("F", "H"), 0.0)
    ))

    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.8) == Set("A", "C", "D", "E", "G", "H"))
    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.2) == Set("A", "B", "C", "D", "E", "G", "H"))
  }

  @Test def ofIBDMatrixTest2() {
    val input: RDD[((String, String), Double)] = sc.parallelize(Array(
      (("A", "B"), .4), (("C", "D"), .3)
    ))

    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.5).size == 2)
  }

}

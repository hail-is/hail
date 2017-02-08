package is.hail.methods

import is.hail.SparkSuite
import org.apache.spark.graphx.{Edge, Graph}
import org.testng.annotations.Test

/**
  * Created by johnc on 2/8/17.
  */
class MaximalIndependentSetSuite extends SparkSuite {

  @Test def graphxIsntDumb() {
    val vertices = sc.parallelize(Array(
      (1L, "A"), (2L, "B"), (3L, "C")
    ))

    val edges = sc.parallelize(Array(
      Edge(1L, 2L, 0), Edge(2L, 3L, 0)
    ))

    val g = Graph(vertices, edges)
    println(g.numEdges)
    println(g.subgraph(_ => true, (id, name) => id != 2L).numEdges)
  }


  @Test def worksAtAll() {
    val input = sc.parallelize(Array(
      (("A", "B"), 0.0), (("B", "C"), 0.0)
    ))

    print(MaximalIndependentSet(sc, input, 0, 1))
  }
}

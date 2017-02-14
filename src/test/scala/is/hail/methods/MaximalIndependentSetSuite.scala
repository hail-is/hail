package is.hail.methods

import is.hail.SparkSuite
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/**
  * Created by johnc on 2/8/17.
  */
class MaximalIndependentSetSuite extends SparkSuite {



  @Test def worksAtAll() {
    val input = sc.parallelize(Array(
      (("A", "B"), 0.2), (("B", "C"), 0.2)
    ))

    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.8) == Set("A", "C"))
    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.8) == Set("A", "C"))
    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.8) == Set("A", "C"))
  }

  @Test def emptySet() {
    val input = sc.parallelize(Array[((String, String), Double)]())

    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, .01) == Set())
  }

  @Test def graphTest1() {
    val input: RDD[((String, String), Double)] = sc.parallelize(Array(
      (("A", "B"), 0.3), (("B", "C"), 0.3), (("B", "D"), 0.3),
      (("D", "F"), 0.0), (("F", "E"), 0.0), (("G", "F"), 0.0),
      (("F", "H"), 0.0)
    ))

    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.8) == Set("A", "C", "D", "E", "G", "H"))
    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.2) == Set("A", "B", "C", "D", "E", "G", "H"))
  }

  @Test def graphTest2() {
    val input: RDD[((String, String), Double)] = sc.parallelize(Array(
      (("A", "B"), .4), (("C", "D"), .3)
    ))

    assert(MaximalIndependentSet.ofIBDMatrix(sc, input, 0.5).size == 2)
  }

}

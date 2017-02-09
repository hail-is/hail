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

    assert(MaximalIndependentSet(sc, input, 0.8, 1) == Set("A", "C"))
    assert(MaximalIndependentSet(sc, input, 0.8, 2) == Set("A", "C"))
    assert(MaximalIndependentSet(sc, input, 0.8, 3) == Set("A", "C"))
  }

  @Test def graphTest1() {
    val input: RDD[((String, String), Double)] = sc.parallelize(Array(
      (("A", "B"), 0.3), (("B", "C"), 0.3), (("B", "D"), 0.3),
      (("D", "F"), 0.0), (("F", "E"), 0.0), (("G", "F"), 0.0),
      (("F", "H"), 0.0)
    ))

    assert(MaximalIndependentSet(sc, input, 0.8, 3) == Set("A", "C", "D", "E", "G", "H"))
    assert(MaximalIndependentSet(sc, input, 0.2, 2) == Set("A", "B", "C", "D", "E", "G", "H"))
  }
}

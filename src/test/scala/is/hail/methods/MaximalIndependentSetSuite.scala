package is.hail.methods

import is.hail.SparkSuite
import org.apache.spark.graphx.{Edge, Graph}
import org.testng.annotations.Test

/**
  * Created by johnc on 2/8/17.
  */
class MaximalIndependentSetSuite extends SparkSuite {


  @Test def worksAtAll() {
    val input = sc.parallelize(Array(
      (("A", "B"), 0.0), (("B", "C"), 0.0)
    ))

    assert(MaximalIndependentSet(sc, input, 0, 1) == Set("A", "C"))
  }
}

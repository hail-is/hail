package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test

class DeNovoSuite extends SparkSuite {

  @Test def test() {
    val kt = hc.read("/Users/tpoterba/data/denovo/TT_esp_ready.vds")
        .filterVariantsExpr("v.start < 825280")
      .filterMulti()
      .deNovo("/Users/tpoterba/data/denovo/TT.fam", "va.esp")

    kt.rdd.collect().foreach(println)
    println(kt.nRows)
  }

}

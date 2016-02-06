package org.broadinstitute.hail.variant

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.methods.LoadVCF
import org.testng.annotations.Test

class HardCallSetSuite extends SparkSuite {
  @Test def test() {
    val d = DenseCallStream.DenseCallStreamFromGtStream(Array(0,1,2,3), 4)

    println(d)

    d.showBinary()

    val vds = LoadVCF(sc, "src/test/resources/linearRegression.vcf")
    val e = HardCallSet(vds)
    println(e.nSamples)
    e.rdd.foreach(println)

    e.write(sqlContext, "/tmp/hardcalls.hcs")

    val f = HardCallSet.read(sqlContext, "/tmp/hardcalls.hcs")

    def toComp(hcs: HardCallSet) = hcs.rdd.mapValues(_.a.toList).collect().toSet

    assert(toComp(e) == toComp(f))
  }
}

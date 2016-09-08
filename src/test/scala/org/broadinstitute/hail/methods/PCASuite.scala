package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

class PCASuite extends SparkSuite {

  @Test def test() {

    val vds = LoadVCF(sc, "src/test/resources/tiny_m.vcf")
    val (scores, loadings, eigenvalues) = (new SamplePCA(3, true, true)) (vds)

    // comparing against numbers computed via Python script test/resources/PCA.py

    val s = scores.toArray
    val s0 = Array(-0.55141958610810227, -0.69169598151052825, 1.487286902938745,
      -0.24417133532011465, 0.64807667470610686, -0.76268431853393925,
      -0.082127077618646974, 0.19673472144647919, -0.3559869584014228,
      -0.13868806289543653, -0.099016362486852916, 0.59369138378371245)

    s.zip(s0).foreach { case (x, y) => assert(D_==(x, y, 1.0E-12)) }


    val l = loadings.get.sortByKey().collect().flatMap { case (v, a) => a }
    val l0 = Array(-0.28047799618430819, 0.41201694824790025, -0.8669337506481809,
      -0.27956988837183494, -0.89909450929475165, -0.33685269907155235,
      0.91824439621061382, -0.14788880184962339, -0.3673637585762754)

    l.zip(l0).foreach { case (x, y) => assert(D_==(x, y, 1.0E-12)) }

    val e = eigenvalues.get
    val e0 = Array(3.0541488634265739, 1.0471401535365061, 0.5082347925607319)

    e.zip(e0).foreach { case (x, y) => assert(D_==(x, y, 1.0E-12)) }

  }
}

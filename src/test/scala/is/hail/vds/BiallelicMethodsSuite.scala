package is.hail.vds

import is.hail.methods.{GRM, VariantQC}
import is.hail.stats.ComputeRRM
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class BiallelicMethodsSuite extends SparkSuite {

  def interceptRequire[T](f: => T) {
    intercept[IllegalArgumentException](f)
  }

  @Test def test() {
    val multi = hc.importVCF("src/test/resources/sample2.vcf")
    val bi = multi.filterMulti()

    interceptRequire {
      multi.concordance(multi)
    }

    interceptRequire {
      multi.concordance(bi)
    }

    interceptRequire {
      bi.concordance(multi)
    }

    interceptRequire {
      multi.exportGen("foo")
    }

    interceptRequire {
      multi.exportPlink("foo")
    }

    interceptRequire {
      multi.ibd()
    }

    interceptRequire {
      GRM(multi)
    }

    interceptRequire {
      multi.mendelErrors(null)
    }

    interceptRequire {
      ComputeRRM(multi)
    }

    interceptRequire {
      multi.imputeSex()
    }

    interceptRequire {
      VariantQC(multi)
    }
  }
}

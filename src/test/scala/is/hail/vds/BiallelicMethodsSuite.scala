package is.hail.vds

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
      multi.grm()
    }

    interceptRequire {
      multi.mendelErrors(null)
    }

    interceptRequire {
      multi.rrm()
    }

    interceptRequire {
      multi.imputeSex()
    }

    interceptRequire {
      multi.pca("foo")
    }

    interceptRequire {
      multi.tdt(null, "foo")
    }

    interceptRequire {
      multi.variantQC()
    }
  }
}

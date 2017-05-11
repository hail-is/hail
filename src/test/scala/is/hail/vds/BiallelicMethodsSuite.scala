package is.hail.vds

import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class BiallelicMethodsSuite extends SparkSuite {

  def catchError[T](f: => T) {
    TestUtils.interceptFatal("requires a split dataset")(f)
  }

  @Test def test() {
    val multi = hc.importVCF("src/test/resources/sample2.vcf")

    catchError {
      multi.concordance(multi)
    }

    catchError {
      multi.exportGen("foo")
    }

    catchError {
      multi.exportPlink("foo")
    }

    catchError {
      multi.ibd()
    }

    catchError {
      multi.grm()
    }

    catchError {
      multi.mendelErrors(null)
    }

    catchError {
      multi.linreg("foo", Array(), "foo", false, 1, 1)
    }

    catchError {
      multi.logreg("foo", "foo", Array(), "foo")
    }

    catchError {
      multi.lmmreg(multi.filterMulti().rrm(true, false), "foo", Array(), true, "foo", "foo", false, None, 1)
    }

    catchError {
      multi.rrm(true, false)
    }

    catchError {
      multi.imputeSex()
    }

    catchError {
      multi.pca("foo")
    }

    catchError {
      multi.tdt(null, "foo")
    }

    catchError {
      multi.variantQC()
    }
  }
}

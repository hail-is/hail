package is.hail.vds

import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class BiallelicMethodsSuite extends SparkSuite {

  def catchError[T](f: => T) {
    TestUtils.interceptUserException("requires a split dataset")(f)
  }

  @Test def test() {
    val multi = hc.importVCF("src/test/resources/sample.vcf")

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
      multi.exportVariantsCassandra("foo", "foo", "foo", "foo", "foo")
    }

    catchError {
      multi.exportVariantsSolr("foo", "foo")
    }

    catchError {
      multi.ibd()
    }

    catchError {
      multi.grm("foo", "foo")
    }

    catchError {
      multi.mendelErrors("foo", "foo")
    }

    catchError {
      multi.linreg("foo", Array(), "foo", 1, 1)
    }

    catchError {
      multi.logreg("foo", "foo", Array(), "foo")
    }

    catchError {
      multi.lmmreg(multi, "foo", Array(), true, "foo", "foo", false, None, 1, true, false)
    }

    catchError {
      multi.imputeSex()
    }

    catchError {
      multi.pca("foo")
    }

    catchError {
      multi.tdt("foo", "foo")
    }

    catchError {
      multi.variantQC()
    }
  }
}

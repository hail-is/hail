package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

class UnicornSuite extends SparkSuite {

  @Test def test() {
    val U = new Unicorn()
    
    val vds = U.alleleCountAnnotate(LoadVCF(sc, "src/test/resources/tiny_m.vcf"))
    
    val var1 = Variant("20", 10019093, "A", "G")
    val var2 = Variant("20", 10026348, "A", "G")
    val var3 = Variant("20", 10026357, "T", "C")

    val (t1, refQuery) = vds.queryVA("va.refCount")
    val (t2, altQuery) = vds.queryVA("va.altCount")
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap
    
    assert(variantAnnotationMap contains var1)
    assert(variantAnnotationMap contains var2)
    assert(variantAnnotationMap contains var3)

    assert(refQuery(variantAnnotationMap(var1)) == Some(5))
    assert(refQuery(variantAnnotationMap(var2)) == Some(7))
    assert(refQuery(variantAnnotationMap(var3)) == Some(6))
    assert(altQuery(variantAnnotationMap(var1)) == Some(1))
    assert(altQuery(variantAnnotationMap(var2)) == Some(1))
    assert(altQuery(variantAnnotationMap(var3)) == Some(2))
    
  }

}

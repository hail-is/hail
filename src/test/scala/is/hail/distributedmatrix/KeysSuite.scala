package is.hail.distributedmatrix

import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.expr.types._
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class KeysSuite extends SparkSuite {  
  val genKeys: Gen[Keys] = for {
      t <- Type.genArb
      v <- TArray(t).genNonmissingValue
    } yield new Keys(t, v.asInstanceOf[IndexedSeq[Annotation]].toArray)
  
  @Test
  def writeReadKeys() {
    val p = forAll(genKeys) { keys =>
      val f = tmpDir.createTempFile()
      keys.write(sc, f)
      keys.assertSame(Keys.read(sc, f))
      true
    }
    
    p.check()
  }

  @Test
  def testFatals() {
    val k1 = new Keys(TInt32(), Array(0, 1))
    val k2 = new Keys(TString(), Array("a", "b"))
    val k3 = new Keys(TInt32(), Array(0, 1, 2))
    val k4 = new Keys(TInt32(), Array(0, 2))
    
    TestUtils.interceptFatal("Keys have different types: Int32, String") {
      k1.assertSame(k2)
    }
    
    TestUtils.interceptFatal("Differing number of keys: 2, 3") {
      k1.assertSame(k3)
    }
    
    TestUtils.interceptFatal("Key mismatch at index 1: 1, 2") {
      k1.assertSame(k4)
    }
  }
  
  @Test
  def filter() {
    val k1 = new Keys(TString(), Array("a", "bc", "d"))
    val k2 = new Keys(TString(), Array("a", "d"))
    
    k1.filter(_.asInstanceOf[String].length < 2).assertSame(k2)
    k1.filter(Array(0, 2)).assertSame(k2)
    
    val (k3, ind) = k1.filterAndIndex(_.asInstanceOf[String].length < 2)
    k2.assertSame(k3)
    assert(Array(0, 2).sameElements(ind))
  }
}

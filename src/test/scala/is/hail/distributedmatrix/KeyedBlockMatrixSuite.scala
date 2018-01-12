package is.hail.distributedmatrix

import is.hail.{SparkSuite, TestUtils}
import is.hail.check.Gen
import is.hail.expr.types._
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class KeyedBlockMatrixSuite extends SparkSuite {
  def assertSame(left: KeyedBlockMatrix, right: KeyedBlockMatrix) {
    (left.rowKeys, right.rowKeys) match {
      case (Some(l), Some(r)) => l.assertSame(r)
      case (None, None) =>
      case _ => assert(false)
    }

    (left.colKeys, right.colKeys) match {
      case (Some(l), Some(r)) => l.assertSame(r)
      case (None, None) =>
      case _ => assert(false)
    }
    
    assert(left.bm.toLocalMatrix() === right.bm.toLocalMatrix())
  }

  @Test
  def writeReadKBM() {
    val nRows = 9
    val nCols = 10
    val lm = Gen.denseMatrix[Double](9, 10).sample()
    val bm = BlockMatrix.from(sc, lm, 3)    
    val rk = new Keys(TInt32(), (0 until nRows).toArray)
    val ck = new Keys(TString(), (0 until nCols).map(_.toString).toArray)
    
    for {
      rowKeys <- Seq(Some(rk), None)
      colKeys <- Seq(Some(ck), None)
    } {
      val file = tmpDir.createTempFile("test")
      val kbm = new KeyedBlockMatrix(bm, rowKeys, colKeys)
      kbm.write(file)
      assertSame(kbm, KeyedBlockMatrix.read(hc, file))
    }
  }

  def fromLocal(lm: BDM[Double], blockSize: Int = 3): KeyedBlockMatrix = {
    val bm = BlockMatrix.from(sc, lm, blockSize)
    val rowKeys = Some(new Keys(TInt32(), (0 until lm.rows).toArray))
    val colKeys = Some(new Keys(TString(), (0 until lm.cols).map(_.toString).toArray))
    new KeyedBlockMatrix(bm, rowKeys, colKeys)
  }  
  
  @Test
  def testOps() {
    val nRows = 9
    val nCols = 10
    val kbm = fromLocal(Gen.denseMatrix[Double](nRows, nCols).sample())
    val zeros = fromLocal(BDM.zeros[Double](nRows, nCols))
    val ones = fromLocal(BDM.ones[Double](nRows, nCols))
    val idLeft = fromLocal(BDM.eye[Double](nRows))
    val idRight = fromLocal(BDM.eye[Double](nCols))
    val keepRows = (0 until nRows).toArray
    val keepCols = (0 until nCols).toArray
    val file = tmpDir.createTempFile("test")
    kbm.write(file)

    for {
      kbm0 <- Seq(
        KeyedBlockMatrix.read(hc, file),
        kbm.add(zeros),
        kbm.add(zeros.dropRowKeys()),
        kbm.add(zeros.dropColKeys()),
        kbm.add(zeros.dropKeys()),
        kbm.dropRowKeys().add(zeros),
        kbm.dropColKeys().add(zeros),
        kbm.dropKeys().add(zeros),
        kbm.dropRowKeys().add(zeros.dropColKeys()),
        kbm.dropColKeys().add(zeros.dropRowKeys()),
        kbm.subtract(zeros),
        kbm.multiply(idRight.dropRowKeys()).setColKeys(kbm.colKeys.get),
        idLeft.setRowKeys(kbm.rowKeys.get).dropColKeys().multiply(kbm),
        kbm.scalarAdd(0),
        kbm.scalarSubtract(0),
        kbm.scalarMultiply(1),
        kbm.scalarDivide(1),
        kbm.pointwiseMultiply(ones),
        kbm.pointwiseDivide(ones),
        kbm.vectorAddToEveryRow(Array.fill[Double](nCols)(0)),
        kbm.vectorAddToEveryColumn(Array.fill[Double](nRows)(0)),
        kbm.vectorPointwiseMultiplyEveryRow(Array.fill[Double](nCols)(1)),
        kbm.vectorPointwiseMultiplyEveryColumn(Array.fill[Double](nRows)(1)),
        kbm.transpose().transpose(),
        kbm.filterRows(_ => true),
        kbm.filterRows(keepRows),
        kbm.filterCols(_ => true),
        kbm.filterCols(keepCols),
        kbm.filter(_ => true, _ => true),
        kbm.filter(keepRows, keepCols),
        kbm.cache(),
        kbm.dropRowKeys().setRowKeys(kbm.rowKeys.get),
        kbm.dropColKeys().setColKeys(kbm.colKeys.get)
      )
    } {
      assertSame(kbm, kbm0)
    }
  }
  
  @Test
  def testFatals() {
    val nRows = 9
    val nCols = 10
    val kbm = fromLocal(Gen.denseMatrix[Double](nRows, nCols).sample())
    
    val strRowKeys = new Keys(TString(), (0 until nRows).map(_.toString).toArray)
    val intColKeys = new Keys(TInt32(), (0 until nCols).toArray)
    
    TestUtils.interceptFatal("Inconsistent row keys. Keys have different types") {
      kbm.add(kbm.setRowKeys(strRowKeys))
    }

    TestUtils.interceptFatal("Inconsistent col keys. Keys have different types") {
      kbm.add(kbm.setColKeys(intColKeys))
    }

    TestUtils.interceptFatal("Differing number of keys and rows") {
      kbm.add(kbm.setRowKeys(intColKeys))
    }
    
    TestUtils.interceptFatal("Differing number of keys and cols") {
      kbm.add(kbm.setColKeys(strRowKeys))
    }
    
    val rowKeys2 = new Keys(TInt32(), (1 until nRows + 1).toArray)
    val colKeys2 = new Keys(TString(), (1 until nCols + 1).map(_.toString).toArray)

    TestUtils.interceptFatal("Inconsistent row keys. Key mismatch at index 0: 0, 1") {
      kbm.add(kbm.setRowKeys(rowKeys2))
    }
    
    TestUtils.interceptFatal("Inconsistent col keys. Key mismatch at index 0: 0, 1") {
      kbm.add(kbm.setColKeys(colKeys2))
    }
    
    val p: Annotation => Boolean = _ => true
    
    TestUtils.interceptFatal("Cannot filter rows using predicate: no row keys") {
      kbm.dropRowKeys().filterRows(p)
    }

    TestUtils.interceptFatal("Cannot filter using predicate: no row keys") {
      kbm.dropRowKeys().filter(p, p)
    }    
    
    TestUtils.interceptFatal("Cannot filter cols using predicate: no col keys") {
      kbm.dropColKeys().filterCols(p)
    }
    
    TestUtils.interceptFatal("Cannot filter using predicate: no col keys") {
      kbm.dropColKeys().filter(p, p)
    }      
    
    TestUtils.interceptFatal("Cannot filter using predicates: no row keys, no col keys") {
      kbm.dropKeys().filter(p, p)
    }
  }
}

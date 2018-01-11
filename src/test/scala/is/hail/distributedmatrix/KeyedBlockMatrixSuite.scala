package is.hail.distributedmatrix

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.expr.types._
import breeze.linalg.{DenseMatrix => BDM}
import org.testng.annotations.Test

class KeyedBlockMatrixSuite extends SparkSuite {
  def assertSame(left: KeyedBlockMatrix, right: KeyedBlockMatrix) {
    (left.rowKeys, right.rowKeys) match {
      case (Some(lrk), Some(rrk)) => lrk.assertSame(rrk)
      case (None, None) =>
      case _ => assert(false)
    }

    (left.colKeys, right.colKeys) match {
      case (Some(lck), Some(rck)) => lck.assertSame(rck)
      case (None, None) =>
      case _ => assert(false)
    }
    
    assert(left.bm.toLocalMatrix() === right.bm.toLocalMatrix())
  }
  
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
}

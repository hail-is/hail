package is.hail.lapack

import com.sun.jna.{Native, Library, Pointer, Memory}
import com.sun.jna.ptr._
import is.hail.annotations.{Memory => HailMemory}

object LAPACKLibrary {
  val instance = Native.loadLibrary("lapack", classOf[LAPACKLibrary]).asInstanceOf[LAPACKLibrary]
}

object BLASLibrary {
  val instance = Native.loadLibrary("blas", classOf[BLASLibrary]).asInstanceOf[BLASLibrary]
}

trait BLASLibrary extends Library {
  //def dgemm(TRANSA: Char, TRANSB: Char, M: Int, N: Int, K: Int, ALPHA: Double, A: Pointer, LDA: Int, B: Pointer, LDB: Int, BETA: Double, C: Pointer, LDC: Int)
  def dgemm(TRANSA: IntByReference, TRANSB: IntByReference, M: IntByReference, N: IntByReference, K: IntByReference,
    ALPHA: DoubleByReference, A: Pointer, LDA: IntByReference, B: Pointer, LDB: IntByReference, BETA: DoubleByReference,
    C: Pointer, LDC: IntByReference)
}

trait LAPACKLibrary extends Library {
  def dsyevd(JOBZ: Char, UPLO: Char, N: Int, A: Pointer, LDA: Int, W: Pointer, WORK: Pointer, LWORK: Int, IWORK: Pointer, LIWORK: Int, INFO: Int)
  //def disnan(DIN: Double)
  def dlapy2(X: DoubleByReference, Y: DoubleByReference): Double
}

object LapackTest {
  final def main(args: Array[String]): Unit = {
    println("Trying to call LAPACK")

    println(LAPACKLibrary.instance)
    println(BLASLibrary.instance)


    val aMem = new Memory(8 * 16)//HailMemory.malloc(8 * 16) //16 8 byte doubles
    val bMem = new Memory (8 * 16) //HailMemory.malloc(8 * 16)
    val cMem = new Memory(8 * 16) //HailMemory.malloc(8 * 16)

    println(aMem.getDouble(0L))

    (0 until 16).foreach{idx =>
      aMem.setDouble(idx.toLong, 1.0)
      bMem.setDouble(idx.toLong, 1.0)
      cMem.setDouble(idx.toLong, 0.0)
    }

    println(aMem.getDouble(0L))
    println(cMem.getDouble(0L))

    val LDA = new IntByReference(4)
    val LDB = new IntByReference(4)
    val LDC = new IntByReference(4)

    BLASLibrary.instance.dgemm(new IntByReference('n'.toInt), new IntByReference('n'),
      new IntByReference(4), new IntByReference(4),
      new IntByReference(4), new DoubleByReference(1.0), aMem,
      LDA, bMem, LDB, new DoubleByReference(1.0),
      cMem, LDC)

    println(aMem.getDouble(0L))
    println(cMem.getDouble(0L))

    //println(LAPACKLibrary.instance.disnan(4.3))
//    var x = 3.0
//    var y = 4.0
//    var z = LAPACKLibrary.instance.dlapy2(new DoubleByReference(x), new DoubleByReference(y))
//
//    println(z)
    //println(res)
    /**
      * Code.invokeStatic[BLAS, BLAS]("getInstance").dgemm("n", "n", getLeftShapeAtIdx(0).toI, getRightShapeAtIdx(1).toI,
      * getLeftShapeAtIdx(1).toI, 1.0, a, getLeftShapeAtIdx(0).toI,
      * b, getRightShapeAtIdx(0).toI, 0, c, getLeftShapeAtIdx(0).toI),
      */

    System.exit(0)
  }
}

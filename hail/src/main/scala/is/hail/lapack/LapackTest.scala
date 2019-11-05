package is.hail.lapack

import com.sun.jna.{Native, Library, Pointer, Memory}
import is.hail.annotations.{Memory => HailMemory}

object LAPACKLibrary {
  val instance = Native.loadLibrary("lapack", classOf[LAPACKLibrary]).asInstanceOf[LAPACKLibrary]
}

object BLASLibrary {
  val instance = Native.loadLibrary("blas", classOf[BLASLibrary]).asInstanceOf[BLASLibrary]
}

trait BLASLibrary extends Library {
  def dgemm(TRANSA: Char, TRANSB: Char, M: Int, N: Int, K: Int, ALPHA: Double, A: Pointer, LDA: Int, B: Pointer, LDB: Int, BETA: Double, C: Pointer, LDC: Int)
}

trait LAPACKLibrary extends Library {
  def dsyevd(JOBZ: Char, UPLO: Char, N: Int, A: Pointer, LDA: Int, W: Pointer, WORK: Pointer, LWORK: Int, IWORK: Pointer, LIWORK: Int, INFO: Int)
}

object LapackTest {
  final def main(args: Array[String]): Unit = {
    println("Trying to call LAPACK")

    println(LAPACKLibrary.instance)
    println(BLASLibrary.instance)


    val aMem = HailMemory.malloc(8 * 16) //16 8 byte doubles
    val bMem = HailMemory.malloc(8 * 16)
    val cMem = HailMemory.malloc(8 * 16)

    BLASLibrary.instance.dgemm('n', 'n', 4, 4, 4, 1.0, new Pointer(aMem), 8, new Pointer(bMem), 8, 1.0, new Pointer(cMem), 8)

    System.exit(0)
  }
}

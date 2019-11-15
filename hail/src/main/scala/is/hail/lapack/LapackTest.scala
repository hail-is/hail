package is.hail.lapack

import com.sun.jna.{Native, Library, Pointer, Memory}
import com.sun.jna.ptr._
import is.hail.annotations.{Memory => HailMemory}
import is.hail.linalg.LAPACKLibrary


object BLASLibrary {
  val instance = Native.loadLibrary("blas", classOf[BLASLibrary]).asInstanceOf[BLASLibrary]
}

trait BLASLibrary extends Library {
  //def dgemm(TRANSA: Char, TRANSB: Char, M: Int, N: Int, K: Int, ALPHA: Double, A: Pointer, LDA: Int, B: Pointer, LDB: Int, BETA: Double, C: Pointer, LDC: Int)
  def dgemm(TRANSA: IntByReference, TRANSB: IntByReference, M: Long, N: Long, K: IntByReference,
    ALPHA: DoubleByReference, A: Long, LDA: IntByReference, B: Long, LDB: IntByReference, BETA: DoubleByReference,
    C: Long, LDC: IntByReference)
}


object LapackTest {
  final def main(args: Array[String]): Unit = {
    println("Trying to call LAPACK")

    println(LAPACKLibrary.getInstance())
//    println(BLASLibrary.instance)

    val major = new IntByReference(0)
    val minor = new IntByReference(0)
    val patch = new IntByReference(0)
    LAPACKLibrary.getInstance().ilaver(major, minor, patch)
    val version = s"${major.getValue}.${minor.getValue}.${patch.getValue}"
    println(version)

    val aMemH = HailMemory.malloc(8 * 16) //16 8 byte doubles
    val bMemH = HailMemory.malloc(8 * 16)
    val cMemH = HailMemory.malloc(8 * 16)

    // Byte 0, M, Byte 4: N, Byte 8: K
    val scratchMemH = HailMemory.malloc(100)

    (0 until 16).foreach{idx =>
      val adjusted = idx.toLong * 8
      HailMemory.storeDouble(aMemH + adjusted, 1.0)
      if (idx % 4 == 0) {
        HailMemory.storeDouble(bMemH + adjusted, 1.0)
      }
      else {
        HailMemory.storeDouble(bMemH + adjusted, 0.0)
      }
      HailMemory.storeDouble(cMemH + adjusted, 0.0)
    }

    println("A")
    (0 until 16).foreach { idx =>
      print(HailMemory.loadDouble(aMemH + idx * 8))
      print(" ")
    }
    println()
    println("B")
    (0 until 16).foreach { idx =>
      print(HailMemory.loadDouble(bMemH + idx * 8))
      print(" ")
    }
    println()
    println(HailMemory.loadDouble(bMemH))
    println(HailMemory.loadDouble(cMemH))
    println("DGEMM")

    val TRANSA = new IntByReference('n'.toInt)
    val TRANSB = new IntByReference('n'.toInt)
    val M = new IntByReference(4)
    val MAddress = scratchMemH
    HailMemory.storeInt(MAddress, 4)
    val N = new IntByReference(4)
    val NAddress = scratchMemH + 4
    HailMemory.storeInt(NAddress, 4)
    val K = new IntByReference(4)
    val KAddress = scratchMemH + 8
    HailMemory.storeInt(KAddress, 4)
    val LDA = new IntByReference(4)
    val LDB = new IntByReference(4)
    val LDC = new IntByReference(4)

    BLASLibrary.instance.dgemm(TRANSA, TRANSB,
      MAddress, NAddress, K, new DoubleByReference(1.0), bMemH,
      LDA, aMemH, LDB, new DoubleByReference(0.0),
      cMemH, LDC)

    println("C")
    (0 until 16).foreach { idx =>
      print(HailMemory.loadDouble(cMemH + idx * 8))
      print(" ")
    }
//
//    println(aMem.getDouble(0L))
//    println(cMem.getDouble(0L))
//    println(cMem.getDouble(8 * 15))


//
//    //println(LAPACKLibrary.instance.disnan(4.3))
////    var x = 3.0
////    var y = 4.0
////    var z = LAPACKLibrary.instance.dlapy2(new DoubleByReference(x), new DoubleByReference(y))
////
////    println(z)
//    //println(res)
//    /**
//      * Code.invokeStatic[BLAS, BLAS]("getInstance").dgemm("n", "n", getLeftShapeAtIdx(0).toI, getRightShapeAtIdx(1).toI,
//      * getLeftShapeAtIdx(1).toI, 1.0, a, getLeftShapeAtIdx(0).toI,
//      * b, getRightShapeAtIdx(0).toI, 0, c, getLeftShapeAtIdx(0).toI),
//      */

    System.exit(0)
  }
}

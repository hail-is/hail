package is.hail.linalg

import com.sun.jna.ptr.{DoubleByReference, IntByReference}


object BLAS {
}

trait BLASLibrary {
  // Not clear on types for TRANSA and TRANSB. Maybe they can be regular chars?
  def dgemm(TRANSA: IntByReference, TRANSB: IntByReference, M: IntByReference, N: IntByReference, K: IntByReference,
    ALPHA: DoubleByReference, A: Long, LDA: IntByReference, B: Long, LDB: IntByReference,
    BETA: DoubleByReference, C: Long, LDC: IntByReference)
}

package is.hail

import breeze.linalg._
import breeze.linalg.operators.{BinaryRegistry, OpMulMatrix}

package object linalg {
  lazy val registerImplOpMulMatrix_DMD_DVD_eq_DVD: Any =
    implicitly[BinaryRegistry[
      DenseMatrix[Double],
      Vector[Double],
      OpMulMatrix.type,
      DenseVector[Double],
    ]].register(
      DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD
    )

}

package is.hail.linalg

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.collection.compat.immutable.ArraySeq

class MatrixSparsitySuite {
  def newToOldReference(from: MatrixSparsity, to: MatrixSparsity): IndexedSeq[Integer] =
    to.definedCoords.map { coords =>
      val i = from.definedCoords.indexOf(coords)
      if (i >= 0) Int.box(i) else null
    }

  def isSubset(s1: MatrixSparsity, s2: MatrixSparsity): Boolean =
    s2.definedCoords.toSet.subsetOf(s1.definedCoords.toSet)

  def sparsities_4_3: Iterator[MatrixSparsity] = Iterator(
    MatrixSparsity.dense(4, 3),
    MatrixSparsity.apply(4, 3, ArraySeq(1 -> 0, 3 -> 0, 2 -> 1, 0 -> 2, 1 -> 2)),
    MatrixSparsity.apply(4, 3, ArraySeq(0 -> 0, 1 -> 0, 0 -> 1, 0 -> 2, 1 -> 2)),
    MatrixSparsity.apply(4, 3, ArraySeq(1 -> 0, 0 -> 2, 1 -> 2)),
    MatrixSparsity.apply(4, 3, ArraySeq()),
  )

  def newToOld() = ArraySeq[(MatrixSparsity, MatrixSparsity.Sparse)](
    (for {
      s1 <- sparsities_4_3
      s2 <- sparsities_4_3
      if s2.isInstanceOf[MatrixSparsity.Sparse]
      if isSubset(s1, s2)
    } yield (s1, s2.asInstanceOf[MatrixSparsity.Sparse])).toSeq: _*
  )

  @ParameterizedTest
  def newToOld(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertEq(s1.newToOldPos(s2), newToOldReference(s1, s2))

  def newToOldNonSubset() = ArraySeq[(MatrixSparsity, MatrixSparsity.Sparse)](
    (for {
      s1 <- sparsities_4_3
      s2 <- sparsities_4_3
      if s2.isInstanceOf[MatrixSparsity.Sparse]
    } yield (s1, s2.asInstanceOf[MatrixSparsity.Sparse])).toSeq: _*
  )

  @ParameterizedTest
  def newToOldNonSubset(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertEq(s1.newToOldPosNonSubset(s2), newToOldReference(s1, s2))

  def sparsities_0_0: Iterator[MatrixSparsity] = Iterator(
    MatrixSparsity.dense(0, 0),
    MatrixSparsity.apply(0, 0, ArraySeq()),
  )

  def degenerateData() = ArraySeq.from(
    sparsities_0_0.map { s1 =>
      (s1, MatrixSparsity.apply(0, 0, ArraySeq()).asInstanceOf[MatrixSparsity.Sparse])
    }
  )

  @ParameterizedTest("degenerateData")
  def newToOldDegenerate(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertEq(s1.newToOldPos(s2), newToOldReference(s1, s2))

  @ParameterizedTest("degenerateData")
  def newToOldNonSubsetDegenerate(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertEq(s1.newToOldPosNonSubset(s2), newToOldReference(s1, s2))
}

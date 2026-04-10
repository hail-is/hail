package is.hail.linalg

import is.hail.TestCaseSupport
import is.hail.collection.compat.immutable.ArraySeq

class MatrixSparsitySuite extends munit.FunSuite with TestCaseSupport {
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

  val sparsitySubsetPairs43: Array[(MatrixSparsity, MatrixSparsity.Sparse)] = (
    for {
      s1 <- sparsities_4_3
      s2 <- sparsities_4_3
      if s2.isInstanceOf[MatrixSparsity.Sparse]
      if isSubset(s1, s2)
    } yield (s1, s2.asInstanceOf[MatrixSparsity.Sparse])
  ).toArray

  val sparsityPairs43: Array[(MatrixSparsity, MatrixSparsity.Sparse)] = (
    for {
      s1 <- sparsities_4_3
      s2 <- sparsities_4_3
      if s2.isInstanceOf[MatrixSparsity.Sparse]
    } yield (s1, s2.asInstanceOf[MatrixSparsity.Sparse])
  ).toArray

  object checkNewToOld extends TestCases {
    def apply(
      s1: MatrixSparsity,
      s2: MatrixSparsity.Sparse,
    )(implicit loc: munit.Location
    ): Unit = test("newToOld") {
      assert(s1.newToOldPos(s2) == newToOldReference(s1, s2))
    }
  }

  sparsitySubsetPairs43.foreach { case (s1, s2) => checkNewToOld(s1, s2) }

  object checkNewToOldNonSubset extends TestCases {
    def apply(
      s1: MatrixSparsity,
      s2: MatrixSparsity.Sparse,
    )(implicit loc: munit.Location
    ): Unit = test("newToOldNonSubset") {
      assert(s1.newToOldPosNonSubset(s2) == newToOldReference(s1, s2))
    }
  }

  sparsityPairs43.foreach { case (s1, s2) => checkNewToOldNonSubset(s1, s2) }

  def sparsities_0_0: Iterator[MatrixSparsity] = Iterator(
    MatrixSparsity.dense(0, 0),
    MatrixSparsity.apply(0, 0, ArraySeq()),
  )

  val sparsityPairs00: Array[(MatrixSparsity, MatrixSparsity.Sparse)] = (
    for {
      s1 <- sparsities_0_0
    } yield (s1, MatrixSparsity.apply(0, 0, ArraySeq()).asInstanceOf[MatrixSparsity.Sparse])
  ).toArray

  object checkNewToOldDegenerate extends TestCases {
    def apply(
      s1: MatrixSparsity,
      s2: MatrixSparsity.Sparse,
    )(implicit loc: munit.Location
    ): Unit = test("newToOldDegenerate") {
      assert(s1.newToOldPos(s2) == newToOldReference(s1, s2))
    }
  }

  sparsityPairs00.foreach { case (s1, s2) => checkNewToOldDegenerate(s1, s2) }

  object checkNewToOldNonSubsetDegenerate extends TestCases {
    def apply(
      s1: MatrixSparsity,
      s2: MatrixSparsity.Sparse,
    )(implicit loc: munit.Location
    ): Unit = test("newToOldNonSubsetDegenerate") {
      assert(s1.newToOldPosNonSubset(s2) == newToOldReference(s1, s2))
    }
  }

  sparsityPairs00.foreach { case (s1, s2) => checkNewToOldNonSubsetDegenerate(s1, s2) }
}

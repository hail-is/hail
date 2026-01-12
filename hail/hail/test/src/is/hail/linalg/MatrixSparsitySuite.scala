package is.hail.linalg

import is.hail.collection.compat.immutable.ArraySeq

import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class MatrixSparsitySuite extends TestNGSuite {
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

  @DataProvider(name = "sparsity_pairs_4_3")
  def sparsityPairs43(): Array[Array[Object]] = (
    for {
      s1 <- sparsities_4_3
      s2 <- sparsities_4_3
      if s2.isInstanceOf[MatrixSparsity.Sparse]
    } yield Array[Object](s1, s2)
  ).toArray

  @DataProvider(name = "sparsity_subset_pairs_4_3")
  def sparsitySubsetPairs(): Array[Array[Object]] = (
    for {
      s1 <- sparsities_4_3
      s2 <- sparsities_4_3
      if s2.isInstanceOf[MatrixSparsity.Sparse]
      if isSubset(s1, s2)
    } yield Array[Object](s1, s2)
  ).toArray

  @Test(dataProvider = "sparsity_subset_pairs_4_3")
  def newToOld(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertResult(newToOldReference(s1, s2))(s1.newToOldPos(s2))

  @Test(dataProvider = "sparsity_pairs_4_3")
  def newToOldNonSubset(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertResult(newToOldReference(s1, s2))(s1.newToOldPosNonSubset(s2))

  def sparsities_0_0: Iterator[MatrixSparsity] = Iterator(
    MatrixSparsity.dense(0, 0),
    MatrixSparsity.apply(0, 0, ArraySeq()),
  )

  @DataProvider(name = "sparsity_pairs_0_0")
  def sparsityPairs00(): Array[Array[Object]] = (
    for {
      s1 <- sparsities_0_0
    } yield Array[Object](s1, MatrixSparsity.apply(0, 0, ArraySeq()))
  ).toArray

  @Test(dataProvider = "sparsity_pairs_0_0")
  def newToOldDegenerate(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertResult(newToOldReference(s1, s2))(s1.newToOldPos(s2))

  @Test(dataProvider = "sparsity_pairs_0_0")
  def newToOldNonSubsetDegenerate(s1: MatrixSparsity, s2: MatrixSparsity.Sparse): Unit =
    assertResult(newToOldReference(s1, s2))(s1.newToOldPosNonSubset(s2))
}

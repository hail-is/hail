package is.hail.utils

import is.hail.TestUtils.assertEq

import org.junit.jupiter.api.Test

class TreeTraversalSuite {

  def binaryTree(i: Int): Iterator[Int] =
    (1 to 2).map(2 * i + _).iterator.filter(_ < 7)

  @Test def testPostOrder() =
    assertEq(
      TreeTraversal.postOrder(binaryTree)(0).toArray,
      Array(3, 4, 1, 5, 6, 2, 0),
    )

  @Test def testPreOrder() =
    assertEq(
      TreeTraversal.preOrder(binaryTree)(0).toArray,
      Array(0, 1, 3, 4, 2, 5, 6),
    )

  @Test def levelOrder() =
    assertEq(
      TreeTraversal.levelOrder(binaryTree)(0).toArray,
      (0 to 6).toArray,
    )

}

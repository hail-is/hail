package is.hail.utils

import org.testng.Assert
import org.testng.annotations.Test

class TreeTraversalSuite {

  def binaryTree(i: Int): Iterator[Int] =
    (1 to 2).map(2 * i + _).iterator.filter(_ < 7)

  @Test def testPostOrder(): Unit =
    Assert.assertEquals(
      TreeTraversal.postOrder(binaryTree)(0).toArray,
      Array(3, 4, 1, 5, 6, 2, 0),
      "",
    )

  @Test def testPreOrder(): Unit =
    Assert.assertEquals(
      TreeTraversal.preOrder(binaryTree)(0).toArray,
      Array(0, 1, 3, 4, 2, 5, 6),
      "",
    )

  @Test def levelOrder(): Unit =
    Assert.assertEquals(
      TreeTraversal.levelOrder(binaryTree)(0).toArray,
      (0 to 6).toArray,
      "",
    )

}

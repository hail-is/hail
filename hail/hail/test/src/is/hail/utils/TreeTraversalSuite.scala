package is.hail.utils

class TreeTraversalSuite extends munit.FunSuite {

  def binaryTree(i: Int): Iterator[Int] =
    (1 to 2).map(2 * i + _).iterator.filter(_ < 7)

  test("PostOrder") {
    assertEquals(
      TreeTraversal.postOrder(binaryTree)(0).toArray.toSeq,
      Array(3, 4, 1, 5, 6, 2, 0).toSeq,
    )
  }

  test("PreOrder") {
    assertEquals(
      TreeTraversal.preOrder(binaryTree)(0).toArray.toSeq,
      Array(0, 1, 3, 4, 2, 5, 6).toSeq,
    )
  }

  test("levelOrder") {
    assertEquals(
      TreeTraversal.levelOrder(binaryTree)(0).toArray.toSeq,
      (0 to 6).toArray.toSeq,
    )
  }

}

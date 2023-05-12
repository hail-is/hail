package is.hail.utils

import scala.collection.mutable

// "Lightweight" (less-safe) implementations of tree traversal algorithms
// inspired by those in Guava
object TreeTraversal {

  def postOrder[A](adj: A => Iterator[A])(root: A): Iterator[A] =
    new Iterator[A] {
      // Java (and Scala) iterators mutate on `next()` so it's convenient
      // to hold on to a node and its children as we visit the node after
      // its children.
      private[this] var stack =
        List((root, adj(root)))

      override def hasNext: Boolean =
        stack.nonEmpty

      override def next(): A = {
        while (stack.head._2.hasNext) {
          val node = stack.head._2.next()
          stack = (node, adj(node)) :: stack
        }

        val (node, _) = stack.head
        stack = stack.tail
        node
      }
    }

  def preOrder[A](adj: A => Iterator[A])(root: A): Iterator[A] =
    new Iterator[A] {
      private[this] var stack =
        List(Iterator.single(root))

      override def hasNext: Boolean =
        stack.nonEmpty

      override def next(): A = {
        val top = stack.head.next()
        if (!stack.head.hasNext)
          stack = stack.tail

        val children = adj(top)
        if (children.hasNext)
          stack = children :: stack

        top
      }
    }

  def levelOrder[A](adj: A => Iterator[A])(root: A): Iterator[A] =
    new Iterator[A] {
      private[this] val queue =
        mutable.Queue(Iterator.single(root))

      override def hasNext: Boolean =
        queue.nonEmpty

      override def next(): A = {
        val top = queue.front.next()
        if (!queue.front.hasNext)
          queue.dequeue()

        val children = adj(top)
        if (children.hasNext)
          queue.enqueue(children)

        top
      }
    }
}

package is.hail.expr.ir

import cats.Monad
import cats.syntax.all._
import is.hail.utils.FastIndexedSeq

import scala.annotation.tailrec
import scala.language.higherKinds

object RewriteBottomUp {

  def apply[M[_]](ir: BaseIR, rule: BaseIR => M[Option[BaseIR]])
                 (implicit M: Monad[M]): M[BaseIR] = {
    val init = (false, List.empty[BaseIR])

    val res =
      IRTraversal.postOrder(ir).foldLeft(M.pure(List.empty[(Boolean, BaseIR)])) {
        (fstack, n) =>
          for {
            stack <- fstack
            ((changed, children), stack_) = dequeue(n.numChildren, init, stack)
            node = if (changed) n.copy(FastIndexedSeq(children: _*)) else n
            rewritten <- rule(node).flatMap {
              case Some(x) => apply(x, rule).map((true, _))
              case None => M.pure((changed, node))
            }
          } yield rewritten :: stack_
      }

    res.map(_.head._2)
  }

  @tailrec
  private def dequeue[A](n: Int, init: (Boolean, List[A]), queue: List[(Boolean, A)])
  : ((Boolean, List[A]), List[(Boolean, A)]) =
    if (n == 0) (init, queue) else {
      val (changed, node) = queue.head
      dequeue(n - 1, (init._1 || changed, node :: init._2), queue.tail)
  }
}

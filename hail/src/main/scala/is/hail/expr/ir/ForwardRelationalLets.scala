package is.hail.expr.ir

import scala.collection.mutable

object ForwardRelationalLets {
  def apply(ir0: BaseIR): BaseIR = {

    val usages = mutable.HashMap.empty[String, (Int, Int)]

    val nestingDepth = NestingDepth(ir0)

    def visit(ir1: BaseIR): Unit = {
      ir1 match {
        case RelationalLet(name, _, _) =>
          usages(name) = (0, 0)
        case RelationalLetTable(name, _, _) =>
          usages(name) = (0, 0)
        case RelationalLetMatrixTable(name, _, _) =>
          usages(name) = (0, 0)
        case RelationalLetBlockMatrix(name, _, _) =>
          usages(name) = (0, 0)
        case x @ RelationalRef(name, _) =>
          val (n, nd) = usages(name)
          usages(name) = (n + 1, math.max(nd, nestingDepth.lookup(x)))
        case _ =>
      }
      ir1.children.foreach(visit)
    }

    visit(ir0)

    def shouldForward(t: (Int, Int)): Boolean = t._1 < 2 && t._2 < 1

    // short circuit if possible
    if (!usages.valuesIterator.exists(shouldForward))
      ir0
    else {
      val m = mutable.HashMap.empty[String, IR]

      def recur(ir1: BaseIR): BaseIR = ir1 match {
        case RelationalLet(name, value, body) =>
          if (shouldForward(usages(name))) {
            m(name) = recur(value).asInstanceOf[IR]
            recur(body)
          } else RelationalLet(name, recur(value).asInstanceOf[IR], recur(body).asInstanceOf[IR])
        case RelationalLetTable(name, value, body) =>
          if (shouldForward(usages(name))) {
            m(name) = recur(value).asInstanceOf[IR]
            recur(body)
          } else RelationalLetTable(
            name,
            recur(value).asInstanceOf[IR],
            recur(body).asInstanceOf[TableIR],
          )
        case RelationalLetMatrixTable(name, value, body) =>
          if (shouldForward(usages(name))) {
            m(name) = recur(value).asInstanceOf[IR]
            recur(body)
          } else RelationalLetMatrixTable(
            name,
            recur(value).asInstanceOf[IR],
            recur(body).asInstanceOf[MatrixIR],
          )
        case RelationalLetBlockMatrix(name, value, body) =>
          if (shouldForward(usages(name))) {
            m(name) = recur(value).asInstanceOf[IR]
            recur(body)
          } else RelationalLetBlockMatrix(
            name,
            recur(value).asInstanceOf[IR],
            recur(body).asInstanceOf[BlockMatrixIR],
          )
        case x @ RelationalRef(name, _) =>
          m.getOrElse(name, x)
        case _ => ir1.mapChildren(recur)
      }

      recur(ir0)
    }
  }
}

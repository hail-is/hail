package is.hail.expr.ir

import scala.collection.mutable

object EvaluateRelationalLets {

  def apply(ir: BaseIR): BaseIR = {

    ExecuteContext.scoped { ctx =>

      def interpretAndCoerce(value: IR): IR = Literal.coerce(value.typ, Interpret(ctx, value, optimize = false))

      def recur(x: BaseIR, m: mutable.Map[String, IR]): BaseIR = {
        x match {
          case RelationalLet(name, value, body) =>
            recur(body, m += (name -> interpretAndCoerce(recur(value, m).asInstanceOf[IR])))
          case RelationalLetTable(name, value, body) =>
            recur(body, m += (name -> interpretAndCoerce(recur(value, m).asInstanceOf[IR])))
          case RelationalLetMatrixTable(name, value, body) =>
            recur(body, m += (name -> interpretAndCoerce(recur(value, m).asInstanceOf[IR])))
          case RelationalLetBlockMatrix(name, value, body) =>
            recur(body, m += (name -> interpretAndCoerce(recur(value, m).asInstanceOf[IR])))
          case RelationalRef(name, t) =>
            m.get(name) match {
              case Some(res) =>
                assert(res.typ == t)
                res
              case None => throw new RuntimeException(name)
            }
          case _ => x.copy(x.children.map(recur(_, m)))
        }
      }

      recur(ir, mutable.HashMap.empty)
    }
  }
}

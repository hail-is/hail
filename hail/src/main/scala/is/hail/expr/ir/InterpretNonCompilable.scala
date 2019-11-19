package is.hail.expr.ir

import is.hail.expr.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable

object InterpretNonCompilable {

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {

    def interpretAndCoerce(value: IR): IR = {
      val preTime = System.nanoTime()
      log.info(s"interpreting non compilable node: ${ value.getClass.getSimpleName }")

      val v = Interpret.alreadyLowered(ctx, value)
      log.info(s"took ${ formatTime(System.nanoTime() - preTime) }")
      if (value.typ == TVoid) {
        Begin(FastIndexedSeq())
      } else Literal.coerce(value.typ, v)
    }

    def rewriteChildren(x: BaseIR, m: mutable.Map[String, IR]): BaseIR = {
      val children = x.children
      val newChildren = children.map(rewrite(_, m))

      // only recons if necessary
      if ((children, newChildren).zipped.forall(_ eq _))
        x
      else
        x.copy(newChildren)
    }


    def rewrite(x: BaseIR, m: mutable.Map[String, IR]): BaseIR = {

      x match {
        case RelationalLet(name, value, body) =>
          rewrite(body, m += (name -> interpretAndCoerce(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetTable(name, value, body) =>
          rewrite(body, m += (name -> interpretAndCoerce(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetMatrixTable(name, value, body) =>
          rewrite(body, m += (name -> interpretAndCoerce(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetBlockMatrix(name, value, body) =>
          rewrite(body, m += (name -> interpretAndCoerce(rewrite(value, m).asInstanceOf[IR])))
        case RelationalRef(name, t) =>
          m.get(name) match {
            case Some(res) =>
              assert(res.typ == t)
              res
            case None => throw new RuntimeException(name)
          }
        case x: IR if InterpretableButNotCompilable(x) => interpretAndCoerce(rewriteChildren(x, m).asInstanceOf[IR])
        case _ => rewriteChildren(x, m)
      }
    }

    rewrite(ir, mutable.HashMap.empty)
  }
}

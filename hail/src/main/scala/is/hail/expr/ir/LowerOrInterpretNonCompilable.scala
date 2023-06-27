package is.hail.expr.ir

import cats.syntax.all._
import is.hail.expr.ir.lowering._
import is.hail.expr.ir.lowering.utils.{assertA, timeM}
import is.hail.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable
import scala.language.higherKinds

object LowerOrInterpretNonCompilable {

  def apply[M[_]: MonadLower](ir: BaseIR): M[BaseIR] = {

    def evaluate(value: IR): M[IR] =
      timeM("LowerOrInterpretNonCompilable/evaluate") {
        for {
          result <- CanLowerEfficiently(value).flatMap {
            case Left(failReason) =>
              log.info(s"LowerOrInterpretNonCompilable: cannot efficiently lower query: $failReason")
              log.info(s"interpreting non-compilable result: ${value.getClass.getSimpleName}")
              Interpret.alreadyLowered(value).map { v =>
                if (value.typ == TVoid) Begin(FastIndexedSeq())
                else Literal.coerce(value.typ, v)
              }
            case Right(()) =>
              log.info(s"LowerOrInterpretNonCompilable: whole stage code generation is a go!")
              log.info(s"lowering result: ${value.getClass.getSimpleName}")
              for {
                fullyLowered <- LowerToDistributedArrayPass(DArrayLowering.All)(value)
                _ = log.info(s"compiling and evaluating result: ${value.getClass.getSimpleName}")
                ir <- CompileAndEvaluate.evalToIR(fullyLowered.asInstanceOf[IR], optimize = true)
              } yield ir
          }
          _ <- assertA(result.typ == value.typ)
        } yield result
      }

    def rewriteChildren(x: BaseIR, m: mutable.Map[String, IR]) =
      x.traverseChildren(rewrite(_, m))

    def rewriteLet(m: mutable.Map[String, IR], name: String, value: IR, body: BaseIR) =
      for {
        rvalue <- rewrite(value, m)
        evald <- evaluate(rvalue.asInstanceOf[IR])
        rewritten <- rewrite(body, m += (name -> evald))
      } yield rewritten

    def rewrite(x: BaseIR, m: mutable.Map[String, IR]): M[BaseIR] =
      x match {
        case RelationalLet(name, value, body) =>
          rewriteLet(m, name, value, body)
        case RelationalLetTable(name, value, body) =>
          rewriteLet(m, name, value, body)
        case RelationalLetMatrixTable(name, value, body) =>
          rewriteLet(m, name, value, body)
        case RelationalLetBlockMatrix(name, value, body) =>
          rewriteLet(m, name, value, body)
        case RelationalRef(name, t) =>
          m.get(name) match {
            case Some(res) =>
              assertA(res.typ == t) *> MonadLower[M].pure(res)
            case None => throw new RuntimeException(name)
          }
        case x: IR if InterpretableButNotCompilable(x) =>
          for { r <- rewriteChildren(x, m); evald <- evaluate(r.asInstanceOf[IR]) }
              yield evald
        case _ => rewriteChildren(x, m)
      }

    rewrite(ir.noSharing, mutable.HashMap.empty)
  }
}

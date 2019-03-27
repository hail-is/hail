package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[Ref]]], defs: Memo[RefEquality[BaseIR]])

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, freeVariablesError: Boolean = true): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[Ref]]]
    val defs = Memo.empty[RefEquality[BaseIR]]

    def computeTable(tir: TableIR): Unit = computeChildren(tir)

    def computeMatrix(mir: MatrixIR): Unit = computeChildren(mir)

    def computeBlockMatrix(bmir: BlockMatrixIR): Unit = computeChildren(bmir)

    def computeChildren(ir0: BaseIR, baseEnv: Option[BindingEnv[RefEquality[BaseIR]]] = None): Unit = {
      ir0.children
        .iterator
        .zipWithIndex
        .foreach {
          case (ir1: IR, i) =>
            val e = ChildEnvWithBindings[RefEquality[BaseIR]](
              ir0,
              i,
              baseEnv.getOrElse(BindingEnv.empty),
              _ => RefEquality(ir0),
              { (b, ab, sb) =>
                if (b.nonEmpty || ab.nonEmpty || sb.nonEmpty && !uses.contains(ir0))
                  uses.bind(ir0, mutable.Set.empty[RefEquality[Ref]])
              }
            )
            computeIR(ir1, e)
          case (tir: TableIR, _) => computeTable(tir)
          case (mir: MatrixIR, _) => computeMatrix(mir)
          case (bmir: BlockMatrixIR, _) => computeBlockMatrix(bmir)
        }
    }

    def computeIR(ir1: BaseIR, env: BindingEnv[RefEquality[BaseIR]]) {
      ir1 match {
        case r@Ref(name, _) =>
          env.eval.lookupOption(name) match {
            case Some(decl) =>
              val re = RefEquality(r)
              uses.lookup(decl) += re
              defs.bind(re, decl)
            case None =>
              if (freeVariablesError)
                throw new RuntimeException(s"found variable with no definition: $name")
          }
        case ir: IR => computeChildren(ir, Some(env))
        case tir: TableIR => computeTable(tir)
        case mir: MatrixIR => computeMatrix(mir)
        case bmir: BlockMatrixIR => computeBlockMatrix(bmir)
      }
    }

    ir0 match {
      case ir: IR => computeIR(ir, BindingEnv.empty)
      case tir: TableIR => computeTable(tir)
      case mir: MatrixIR => computeMatrix(mir)
      case bmir: BlockMatrixIR => computeBlockMatrix(bmir)
    }

    UsesAndDefs(uses, defs)
  }
}

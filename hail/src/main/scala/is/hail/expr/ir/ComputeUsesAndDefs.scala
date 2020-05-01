package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[BaseRef]]], defs: Memo[BaseIR])

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, errorIfFreeVariables: Boolean = true, includeApplyIR: Boolean = false): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[BaseRef]]]
    val defs = Memo.empty[BaseIR]

    def computeTable(tir: TableIR, includeApplyIR: Boolean): Unit = tir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(tir, i).mapValues[BaseIR](_ => tir)
          if (!b.allEmpty && !uses.contains(tir))
            uses.bind(tir, mutable.Set.empty[RefEquality[BaseRef]])
          computeIR(child, b, includeApplyIR)
        case (child: TableIR, _) => computeTable(child, includeApplyIR)
        case (child: MatrixIR, _) => computeMatrix(child, includeApplyIR)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child, includeApplyIR)
      }

    def computeMatrix(mir: MatrixIR, includeApplyIR: Boolean): Unit = mir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(mir, i).mapValues[BaseIR](_ => mir)
          if (!b.allEmpty && !uses.contains(mir))
            uses.bind(mir, mutable.Set.empty[RefEquality[BaseRef]])
          computeIR(child, b, includeApplyIR)
        case (child: TableIR, _) => computeTable(child, includeApplyIR)
        case (child: MatrixIR, _) => computeMatrix(child, includeApplyIR)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child, includeApplyIR)
      }

    def computeBlockMatrix(bmir: BlockMatrixIR, includeApplyIR: Boolean): Unit = bmir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(bmir, i).mapValues[BaseIR](_ => bmir)
          if (!b.allEmpty && !uses.contains(bmir))
            uses.bind(bmir, mutable.Set.empty[RefEquality[BaseRef]])
          computeIR(child, b, includeApplyIR)
        case (child: TableIR, _) => computeTable(child, includeApplyIR)
        case (child: MatrixIR, _) => computeMatrix(child, includeApplyIR)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child, includeApplyIR)
      }

    def computeIR(ir: IR, env: BindingEnv[BaseIR], includeApplyIR: Boolean) {
      ir match {
        case r: BaseRef =>
          env.eval.lookupOption(r.name) match {
            case Some(decl) =>
              val re = RefEquality(r)
              uses.lookup(decl) += re
              defs.bind(re, decl)
            case None =>
              if (errorIfFreeVariables)
                throw new RuntimeException(s"found variable with no definition: ${ r.name }")
          }
        case _ =>
      }
      ir match {
        case x: ApplyIR if includeApplyIR => computeIR(x.explicitNode, env, includeApplyIR)
        case _: IR =>
          ir.children.iterator.zipWithIndex
            .foreach {
              case (ir1: IR, i) =>
                val e = ChildEnvWithoutBindings(ir, i, env)
                val newBindings = NewBindings(ir, i, e)

                if (newBindings.allEmpty)
                  computeIR(ir1, e, includeApplyIR)
                else {
                  if (!uses.contains(ir))
                    uses.bind(ir, mutable.Set.empty[RefEquality[BaseRef]])
                  computeIR(ir1, e.merge(newBindings.mapValues(_ => ir)), includeApplyIR)
                }
              case (tir: TableIR, _) => computeTable(tir, includeApplyIR)
              case (mir: MatrixIR, _) => computeMatrix(mir, includeApplyIR)
              case (bmir: BlockMatrixIR, _) => computeBlockMatrix(bmir, includeApplyIR)
            }
      }
    }

    ir0 match {
      case ir: IR => computeIR(ir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)), includeApplyIR)
      case tir: TableIR => computeTable(tir, includeApplyIR)
      case mir: MatrixIR => computeMatrix(mir, includeApplyIR)
      case bmir: BlockMatrixIR => computeBlockMatrix(bmir, includeApplyIR)
    }

    UsesAndDefs(uses, defs)
  }
}

package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[Ref]]], defs: Memo[BaseIR])

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, allowFreeVariables: Boolean = true): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[Ref]]]
    val defs = Memo.empty[BaseIR]

    def computeTable(tir: TableIR): Unit = tir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(tir, i).mapValues[BaseIR](_ => tir)
          if (!b.allEmpty && !uses.contains(tir))
            uses.bind(tir, mutable.Set.empty[RefEquality[Ref]])
          computeIR(child, b)
        case (child: TableIR, _) => computeTable(child)
        case (child: MatrixIR, _) => computeMatrix(child)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child)
      }

    def computeMatrix(mir: MatrixIR): Unit = mir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(mir, i).mapValues[BaseIR](_ => mir)
          if (!b.allEmpty && !uses.contains(mir))
            uses.bind(mir, mutable.Set.empty[RefEquality[Ref]])
          computeIR(child, b)
        case (child: TableIR, _) => computeTable(child)
        case (child: MatrixIR, _) => computeMatrix(child)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child)
      }

    def computeBlockMatrix(bmir: BlockMatrixIR): Unit = bmir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(bmir, i).mapValues[BaseIR](_ => bmir)
          if (!b.allEmpty && !uses.contains(bmir))
            uses.bind(bmir, mutable.Set.empty[RefEquality[Ref]])
          computeIR(child, b)
        case (child: TableIR, _) => computeTable(child)
        case (child: MatrixIR, _) => computeMatrix(child)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child)
      }

    def computeIR(ir: IR, env: BindingEnv[BaseIR]) {
      ir match {
        case r@Ref(name, _) =>
          env.eval.lookupOption(name) match {
            case Some(decl) =>
              val re = RefEquality(r)
              uses.lookup(decl) += re
              defs.bind(re, decl)
            case None =>
              if (allowFreeVariables)
                throw new RuntimeException(s"found variable with no definition: $name")
          }
        case _: IR =>
          ir.children
            .iterator
            .zipWithIndex
            .foreach {
              case (ir1: IR, i) =>
                val e = ChildEnvWithoutBindings(ir, i, env)
                val newBindings = NewBindings(ir, i, e)

                if (newBindings.allEmpty)
                  computeIR(ir1, e)
                else {
                  if (!uses.contains(ir))
                    uses.bind(ir, mutable.Set.empty[RefEquality[Ref]])
                  computeIR(ir1, e.merge(newBindings.mapValues(_ => ir)))
                }
              case (tir: TableIR, _) => computeTable(tir)
              case (mir: MatrixIR, _) => computeMatrix(mir)
              case (bmir: BlockMatrixIR, _) => computeBlockMatrix(bmir)
            }
      }
    }

    ir0 match {
      case ir: IR => computeIR(ir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)))
      case tir: TableIR => computeTable(tir)
      case mir: MatrixIR => computeMatrix(mir)
      case bmir: BlockMatrixIR => computeBlockMatrix(bmir)
    }

    UsesAndDefs(uses, defs)
  }
}

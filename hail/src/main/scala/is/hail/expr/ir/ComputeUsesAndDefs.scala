package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[Ref]]], defs: Memo[RefEquality[BaseIR]])

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, freeVariablesError: Boolean = true): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[Ref]]]
    val defs = Memo.empty[RefEquality[BaseIR]]

    def computeTable(tir: TableIR): Unit = tir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val b = NewBindings(tir, i).mapValues[RefEquality[BaseIR]](_ => RefEquality(tir))
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
          val b = NewBindings(mir, i).mapValues[RefEquality[BaseIR]](_ => RefEquality(mir))
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
          val b = NewBindings(bmir, i).mapValues[RefEquality[BaseIR]](_ => RefEquality(bmir))
          if (!b.allEmpty && !uses.contains(bmir))
            uses.bind(bmir, mutable.Set.empty[RefEquality[Ref]])
          computeIR(child, b)
        case (child: TableIR, _) => computeTable(child)
        case (child: MatrixIR, _) => computeMatrix(child)
        case (child: BlockMatrixIR, _) => computeBlockMatrix(child)
      }

    def computeIR(ir: IR, env: BindingEnv[RefEquality[BaseIR]]) {
      ir match {
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
                  val re = RefEquality(ir)
                  if (!uses.contains(re))
                    uses.bind(re, mutable.Set.empty[RefEquality[Ref]])
                  computeIR(ir1, e.merge(newBindings.mapValues(_ => re)))
                }
              case (tir: TableIR, _) => computeTable(tir)
              case (mir: MatrixIR, _) => computeMatrix(mir)
              case (bmir: BlockMatrixIR, _) => computeBlockMatrix(bmir)
            }
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

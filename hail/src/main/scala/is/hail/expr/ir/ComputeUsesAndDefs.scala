package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[BaseRef]]], defs: Memo[BaseIR], free: mutable.Set[RefEquality[BaseRef]])

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, errorIfFreeVariables: Boolean = true): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[BaseRef]]]
    val defs = Memo.empty[BaseIR]
    val free = if (errorIfFreeVariables) null else mutable.Set[RefEquality[BaseRef]]()

    def computeTable(tir: TableIR, env: BindingEnv[BaseIR]): Unit = tir.children
      .iterator
      .zipWithIndex
      .foreach { case (child, i) =>
        val e = ChildEnvWithoutBindings(tir, i, env)
        val b = NewBindings(tir, i).mapValues[BaseIR](_ => tir)
        if (!b.allEmpty && !uses.contains(tir))
          uses.bind(tir, mutable.Set.empty[RefEquality[BaseRef]])
        val childEnv = e.merge(b)
        child match {
          case child: IR => computeIR(child, childEnv)
          case child: TableIR => computeTable(child, childEnv)
          case child: MatrixIR => computeMatrix(child, childEnv)
          case child: BlockMatrixIR => computeBlockMatrix(child, childEnv)
        }
      }

    def computeMatrix(mir: MatrixIR, env: BindingEnv[BaseIR]): Unit = mir.children
      .iterator
      .zipWithIndex
      .foreach { case (child, i) =>
        val e = ChildEnvWithoutBindings(mir, i, env)
        val b = NewBindings(mir, i).mapValues[BaseIR](_ => mir)
        if (!b.allEmpty && !uses.contains(mir))
          uses.bind(mir, mutable.Set.empty[RefEquality[BaseRef]])
        val childEnv = e.merge(b)
        child match {
          case child: IR => computeIR(child, childEnv)
          case child: TableIR => computeTable(child, childEnv)
          case child: MatrixIR => computeMatrix(child, childEnv)
          case child: BlockMatrixIR => computeBlockMatrix(child, childEnv)
        }
      }

    def computeBlockMatrix(bmir: BlockMatrixIR, env: BindingEnv[BaseIR]): Unit = bmir.children
      .iterator
      .zipWithIndex
      .foreach { case (child, i) =>
        val e = ChildEnvWithoutBindings(bmir, i, env)
        val b = NewBindings(bmir, i).mapValues[BaseIR](_ => bmir)
        if (!b.allEmpty && !uses.contains(bmir))
          uses.bind(bmir, mutable.Set.empty[RefEquality[BaseRef]])
        val childEnv = e.merge(b)
        child match {
          case child: IR => computeIR(child, childEnv)
          case child: TableIR => computeTable(child, childEnv)
          case child: MatrixIR => computeMatrix(child, childEnv)
          case child: BlockMatrixIR => computeBlockMatrix(child, childEnv)
        }
      }

    def computeIR(ir: IR, env: BindingEnv[BaseIR]): Unit = {

      ir match {
        case r: BaseRef =>
          val e = r match {
            case _: Ref => env.eval
            case _: Recur => env.eval
            case _: RelationalRef => env.relational
          }
          e.lookupOption(r.name) match {
            case Some(decl) =>
              if (!defs.contains(r)) {
                val re = RefEquality(r)
                uses.lookup(decl) += re
                defs.bind(re, decl)
              }
            case None =>
              if (errorIfFreeVariables)
                throw new RuntimeException(s"found variable with no definition: ${ r.name }")
              else
                free += RefEquality(r)
          }
        case _ =>
      }

      ir.children
        .iterator
        .zipWithIndex
        .foreach { case (child, i) =>
          val e = ChildEnvWithoutBindings(ir, i, env)
          val b = NewBindings(ir, i, env).mapValues[BaseIR](_ => ir)
          if (!b.allEmpty && !uses.contains(ir))
            uses.bind(ir, mutable.Set.empty[RefEquality[BaseRef]])
          val childEnv = e.merge(b)
          child match {
            case child: IR => computeIR(child, childEnv)
            case child: TableIR => computeTable(child, childEnv)
            case child: MatrixIR => computeMatrix(child, childEnv)
            case child: BlockMatrixIR => computeBlockMatrix(child, childEnv)
          }
        }
    }
    val startE = BindingEnv[BaseIR](Env.empty, Some(Env.empty), Some(Env.empty))
    ir0 match {
      case ir: IR => computeIR(ir, startE)
      case tir: TableIR => computeTable(tir, startE)
      case mir: MatrixIR => computeMatrix(mir, startE)
      case bmir: BlockMatrixIR => computeBlockMatrix(bmir, startE)
    }

    UsesAndDefs(uses, defs, free)
  }
}

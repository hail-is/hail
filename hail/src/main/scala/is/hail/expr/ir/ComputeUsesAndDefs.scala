package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[BaseRef]]], defs: Memo[BaseIR], free: mutable.Set[RefEquality[BaseRef]])

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, errorIfFreeVariables: Boolean = true): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[BaseRef]]]
    val defs = Memo.empty[BaseIR]
    val free = if (errorIfFreeVariables) null else mutable.Set[RefEquality[BaseRef]]()

    def computeTable(tir: TableIR, env: Env[BaseIR]): Unit = {
      var i = 0
      while (i < tir.children.length) {
        val b = NewBindings(tir, i).mapValues[BaseIR](_ => tir)
          .bindDefinedScopes(env.m.toArray: _*)
        if (!b.allEmpty && !uses.contains(tir))
          uses.bind(tir, mutable.Set.empty[RefEquality[BaseRef]])
        tir.children(i) match {
          case child: IR => computeIR(child, b)
          case child: TableIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeTable(child, b.eval)
          case child: MatrixIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeMatrix(child, b.eval)
          case child: BlockMatrixIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeBlockMatrix(child, b.eval)
        }
        i += 1
      }
    }

    def computeMatrix(mir: MatrixIR, env: Env[BaseIR]): Unit = {
      var i = 0
      while (i < mir.children.length) {
        val b = NewBindings(mir, i).mapValues[BaseIR](_ => mir)
          .bindDefinedScopes(env.m.toArray: _*)
        if (!b.allEmpty && !uses.contains(mir))
          uses.bind(mir, mutable.Set.empty[RefEquality[BaseRef]])
        mir.children(i) match {
          case child: IR => computeIR(child, b)
          case child: TableIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeTable(child, b.eval)
          case child: MatrixIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeMatrix(child, b.eval)
          case child: BlockMatrixIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeBlockMatrix(child, b.eval)
        }
        i += 1
      }
    }

    def computeBlockMatrix(bmir: BlockMatrixIR, env: Env[BaseIR]): Unit = {
      var i = 0
      while (i < bmir.children.length) {
        val b = NewBindings(bmir, i).mapValues[BaseIR](_ => bmir)
          .bindDefinedScopes(env.m.toArray: _*)
        if (!b.allEmpty && !uses.contains(bmir))
          uses.bind(bmir, mutable.Set.empty[RefEquality[BaseRef]])
        bmir.children(i) match {
          case child: IR => computeIR(child, b)
          case child: TableIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeTable(child, b.eval)
          case child: MatrixIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeMatrix(child, b.eval)
          case child: BlockMatrixIR =>
            assert(b.agg.isEmpty && b.scan.isEmpty)
            computeBlockMatrix(child, b.eval)
        }
        i += 1
      }
    }

    def computeIR(ir: IR, env: BindingEnv[BaseIR]) {
      ir match {
        case r: BaseRef =>
          env.eval.lookupOption(r.name) match {
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
      ir.children.iterator.zipWithIndex
        .foreach {
          case (ir1: IR, i) =>
            val e = ChildEnvWithoutBindings(ir, i, env)
            val newBindings = NewBindings(ir, i, e)

            if (newBindings.allEmpty)
              computeIR(ir1, e)
            else {
              if (!uses.contains(ir))
                uses.bind(ir, mutable.Set.empty[RefEquality[BaseRef]])
              computeIR(ir1, e.merge(newBindings.mapValues(_ => ir)))
            }
              case (tir: TableIR, _) => computeTable(tir, Env.empty)
              case (mir: MatrixIR, _) => computeMatrix(mir, Env.empty)
              case (bmir: BlockMatrixIR, _) => computeBlockMatrix(bmir, Env.empty)
            }
    }

    ir0 match {
      case ir: IR => computeIR(ir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)))
      case tir: TableIR => computeTable(tir, Env.empty)
      case mir: MatrixIR => computeMatrix(mir, Env.empty)
      case bmir: BlockMatrixIR => computeBlockMatrix(bmir, Env.empty)
    }

    UsesAndDefs(uses, defs, free)
  }
}

package is.hail.expr.ir

import is.hail.utils._

import scala.collection.mutable

object ForwardLets {
  def apply(ir0: BaseIR): BaseIR = {
    val ir1 = new NormalizeNames(allowFreeVariables = true).apply(ir0)
    val UsesAndDefs(uses, _) = ComputeUsesAndDefs(ir1, allowFreeVariables = false)
    val nestingDepth = NestingDepth(ir1)

    def rewriteTable(tir: TableIR): BaseIR = tir.copy(tir
      .children
      .iterator
      .zipWithIndex
      .map {
      case (child: IR, i) => rewrite(child, NewBindings(tir, i).dropBindings)
      case (tir: TableIR, _) => rewriteTable(tir)
      case (mir: MatrixIR, _) => rewriteMatrix(mir)
      case (bmir: BlockMatrixIR, _) => rewriteBlockMatrix(bmir)
    }.toFastIndexedSeq)

    def rewriteMatrix(mir: MatrixIR): BaseIR = mir.copy(mir
      .children
      .iterator
      .zipWithIndex
      .map {
        case (child: IR, i) => rewrite(child, NewBindings(mir, i).dropBindings)
        case (tir: TableIR, _) => rewriteTable(tir)
        case (mir: MatrixIR, _) => rewriteMatrix(mir)
        case (bmir: BlockMatrixIR, _) => rewriteBlockMatrix(bmir)
      }.toFastIndexedSeq)

    def rewriteBlockMatrix(bmir: BlockMatrixIR): BaseIR = bmir.copy(bmir
      .children
      .iterator
      .zipWithIndex
      .map {
        case (child: IR, i) => rewrite(child, NewBindings(bmir, i).dropBindings)
        case (tir: TableIR, _) => rewriteTable(tir)
        case (mir: MatrixIR, _) => rewriteMatrix(mir)
        case (bmir: BlockMatrixIR, _) => rewriteBlockMatrix(bmir)
      }.toFastIndexedSeq)

    def rewrite(ir: IR, env: BindingEnv[IR]): IR = {

      def shouldForward(value: IR, refs: mutable.Set[RefEquality[Ref]], base: IR): Boolean = {
        value.isInstanceOf[Ref] ||
          IsConstant(value) ||
          refs.isEmpty ||
          (refs.size == 1 &&
            nestingDepth.lookup(refs.head) == nestingDepth.lookup(base) &&
            !ContainsScan(value) &&
            !ContainsAgg(value))
      }

      def mapRewrite(): IR = ir.copy(ir.children
        .iterator
        .zipWithIndex
        .map {
          case (ir1: IR, i) => rewrite(ir1, ChildEnvWithoutBindings(ir, i, env))
          case (tir: TableIR, _) => rewriteTable(tir)
          case (mir: MatrixIR, _) => rewriteMatrix(mir)
          case (bmir: BlockMatrixIR, _) => rewriteBlockMatrix(bmir)
        }.toFastIndexedSeq)

      ir match {
        case Let(name, value, body) =>
          val refs = uses.lookup(ir)
          if (shouldForward(value, refs, ir))
            rewrite(body, env.bindEval(name -> rewrite(value, env)))
          else
            mapRewrite()
        case AggLet(name, value, body, isScan) =>
          val refs = uses.lookup(ir)
          if (shouldForward(value, refs, ir))
            if (isScan)
              rewrite(body, env.copy(scan = Some(env.scan.get.bind(name -> rewrite(value, BindingEnv(env.scanOrEmpty))))))
            else
              rewrite(body, env.copy(agg = Some(env.agg.get.bind(name -> rewrite(value, BindingEnv(env.aggOrEmpty))))))
          else
            mapRewrite()
        case x@Ref(name, _) => env.eval.lookupOption(name).getOrElse(x)
        case _ =>
          mapRewrite()
      }
    }

    ir1 match {
      case ir: IR => rewrite(ir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)))
      case tir: TableIR => rewriteTable(tir)
      case mir: MatrixIR => rewriteMatrix(mir)
      case bmir: BlockMatrixIR => rewriteBlockMatrix(bmir)
    }
  }
}

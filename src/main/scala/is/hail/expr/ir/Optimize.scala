package is.hail.expr.ir

import is.hail.expr._

object Optimize {
  private def optimize(ir: BaseIR): BaseIR = {
    RewriteBottomUp(ir, {
      // optimize TableIR
      case TableFilter(t, True()) => t

      case TableFilter(TableRead(path, spec, _), False() | NA(_)) =>
        TableRead(path, spec, dropRows = true)

      case TableFilter(TableFilter(t, p1), p2) =>
        TableFilter(t,
          ApplySpecial("&&", Array(p1, p2)))

      // optimize MatrixIR

      // Equivalent rewrites for the new Filter{Cols,Rows}IR
      case FilterRowsIR(MatrixRead(path, spec, dropSamples, _), False() | NA(_)) =>
        MatrixRead(path, spec, dropSamples, dropRows = true)

      case FilterColsIR(MatrixRead(path, spec, dropVariants, _), False() | NA(_)) =>
        MatrixRead(path, spec, dropCols = true, dropVariants)

      // Keep all rows/cols = do nothing
      case FilterRowsIR(m, True()) => m

      case FilterColsIR(m, True()) => m

      // Push FilterRowsIR into FilterColsIR
      case FilterRowsIR(FilterColsIR(m, colPred), rowPred) =>
        FilterColsIR(FilterRowsIR(m, rowPred), colPred)

      // Combine multiple filters into one
      case FilterRowsIR(FilterRowsIR(m, pred1), pred2) =>
        FilterRowsIR(m,
          ApplySpecial("&&", Array(pred1, pred2)))

      case FilterColsIR(FilterColsIR(m, pred1), pred2) =>
        FilterColsIR(m,
          ApplySpecial("&&", Array(pred1, pred2)))
    })
  }

  def apply(ir: TableIR): TableIR = optimize(ir).asInstanceOf[TableIR]

  def apply(ir: MatrixIR): MatrixIR = optimize(ir).asInstanceOf[MatrixIR]

  def apply(ir: IR): IR = optimize(ir).asInstanceOf[IR]
}

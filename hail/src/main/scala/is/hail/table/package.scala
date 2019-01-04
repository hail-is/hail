package is.hail

import is.hail.expr.ir.Sym

package object table {
  def asc(field: Sym): SortField = SortField(field, Ascending)

  def desc(field: Sym): SortField = SortField(field, Descending)
}

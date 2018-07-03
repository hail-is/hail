package is.hail

package object table {
  def asc(field: String): SortField = SortField(field, Ascending)

  def desc(field: String): SortField = SortField(field, Descending)
}

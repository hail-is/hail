package is.hail

package object keytable {
  def asc(field: String): SortColumn = SortColumn(field, Ascending)

  def desc(field: String): SortColumn = SortColumn(field, Descending)
}

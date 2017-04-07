package is.hail

package object keytable {
  def asc(field: String): SortField = SortField(field, Ascending)

  def desc(field: String): SortField = SortField(field, Descending)
}

package is.hail.utils

case class TextInputFilterAndReplace(
  filterPattern: Option[String] = None,
  findPattern: Option[String] = None,
  replacePattern: Option[String] = None,
) {
  require(!(findPattern.isDefined ^ replacePattern.isDefined))

  private val fpRegex = filterPattern.map(_.r).orNull
  private val find = findPattern.orNull
  private val replace = replacePattern.orNull

  def apply(it: Iterator[WithContext[String]]): Iterator[WithContext[String]] = {
    var iter = it
    if (fpRegex != null)
      iter = iter.filter(c => fpRegex.findFirstIn(c.value).isEmpty)
    if (find != null)
      iter = iter.map(c => c.map(_.replaceAll(find, replace)))
    iter
  }

  def transformer(): String => String = {
    if (fpRegex != null && find != null) {
      (s: String) =>
        if (fpRegex.findFirstIn(s).isEmpty)
          null
        else
          s.replaceAll(find, replace)
    } else if (fpRegex != null) {
      (s: String) =>
        if (fpRegex.findFirstIn(s).isEmpty)
          s
        else
          null
    } else if (find != null) {
      (s: String) => s.replaceAll(find, replace)
    } else
      (s: String) => s
  }
}

package is.hail.utils

case class TextInputFilterAndReplace(filterPattern: Option[String] = None, findPattern: Option[String] = None, replacePattern: Option[String] = None) {
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
}

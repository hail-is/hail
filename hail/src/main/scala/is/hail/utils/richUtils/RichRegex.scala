package is.hail.utils.richUtils

import scala.util.matching.Regex

class RichRegex(r: Regex) {
  def matches(s: String): Boolean = r.pattern.matcher(s).matches()
}

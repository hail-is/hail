package is.hail.utils.richUtils

import is.hail.utils.Truncatable

class RichString(val str: String) extends AnyVal {
  def truncatable(toTake: Int = 60): Truncatable = new Truncatable {
    override def truncate: String = if (str.length > toTake - 3) str.take(toTake) + "..." else str

    override def strings: (String, String) = (truncate, str)
  }

  def equalsCaseInsensitive(other: String): Boolean =
    if (str.length == other.length) {
      var i = 0
      while (i < str.length) {
        if ((str charAt i).toLower != (other charAt i).toLower)
          return false
        i += 1
      }
      true
    } else false
}

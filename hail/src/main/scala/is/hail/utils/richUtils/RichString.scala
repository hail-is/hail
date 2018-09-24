package is.hail.utils.richUtils

import is.hail.utils.Truncatable

class RichString(val str: String) extends AnyVal {
  def truncatable(toTake: Int = 60): Truncatable = new Truncatable {
    def truncate: String = if (str.length > toTake - 3) str.take(toTake) + "..." else str

    def strings: (String, String) = (truncate, str)
  }
}

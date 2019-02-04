package is.hail.nativecode

class PrettyCode(in: String) {
  var numLines = 0
  val outStr = filter(in)
  
  private def filter(s: String): String = {
    val kNewline = '\n'
    val kDQuote = '\"'
    val kBackslash = '\\'
    val kBra = '('
    val kKet = ')'
    val kCurlyBra = '{'
    val kCurlyKet = '}'
    val kSpace = ' '
    val kTab = '\t'

    val len = s.length
    val sb = new StringBuilder(len*2)
    var state = 0
    var depth = 0
    var pos = 0
    while (pos < len) {
      val c = s.charAt(pos)
      pos += 1
      if (state <= 1) {
        if ((c == kKet) || (c == kCurlyKet)) depth -= 1;
        if (depth < 0) depth = 0
      }
      if (c == kNewline) {
        if (state != 0) {
          numLines += 1
          sb.append(c)
          state = 0
        }
      } else if (state == 0) { // start of line
        if ((c == kSpace) || (c == kTab)) {
          // skip
        } else {
          var j = 0;
          while (j < depth) {
            sb.append("  ")
            j += 1
          }
          sb.append(c)
          state = if (c == kDQuote) 2 else 1
        }
      } else if (state == 1) { // normal
        sb.append(c)
      } else if (state == 2) { // in quotes
        sb.append(c)
        if      (c == kBackslash) state = 3
        else if (c == kDQuote)    state = 1
      } else {                 // in quotes after backslash
        sb.append(c)
        state = 2
      }
      if (state == 1) {
        if ((c == kBra) || (c == kCurlyBra)) depth += 1
      }
    }
    sb.toString()
  }
  
  override def toString(): String = outStr
  
  def countLines(): Int = numLines
}

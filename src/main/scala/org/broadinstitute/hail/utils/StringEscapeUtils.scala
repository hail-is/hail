package org.broadinstitute.hail.utils

import java.util.Locale

import org.broadinstitute.hail.utils._

object StringEscapeUtils {

  def hex(ch: Char): String = Integer.toHexString(ch).toUpperCase(Locale.ENGLISH)

  def escapeStringSimple(str: String, escapeChar: Char,
    escapeFirst: (Char) => Boolean,
    escape: (Char) => Boolean): String = {
    val sb = new StringBuilder
    var i: Int = 0
    while (i < str.length) {
      val c = str(i)
      if (c == escapeChar || escape(c) || (i == 0 && escapeFirst(c))) {
        val i = c.toInt
        if (i < 256) {
          val h = Integer.toHexString(i)
          assert(h.length == 2)
          sb += escapeChar
          sb.append(h)
        } else {

        }
      } else
        sb += c
      i += 1
    }
    sb.result()
  }

  def escapeStringSimple(str: String, escapeChar: Char, escape: (Char) => Boolean): String
   = escapeStringSimple(str, escapeChar, escape, escape)

  def unescapeStringSimple(str: String, escapeChar: Char): String = {
    val sb = new StringBuilder
    var i: Int = 0
    while (i < str.length) {
      val c = str(i)
      if (c == escapeChar) {
        sb += Integer.parseInt(str.substring(i + 1, i + 3), 16).toChar
        i += 3
      } else {
        sb += c
        i += 1
      }
    }
    sb.result()
  }

  def escapeString(str: String, backticked: Boolean = false): String =
    escapeString(str, new StringBuilder(capacity = str.length * 2), backticked)

  def escapeString(str: String, sb: StringBuilder, backticked: Boolean): String = {
    sb.clear()

    var sz: Int = 0
    sz = str.length
    var i: Int = 0
    while (i < sz) {
      val ch: Char = str.charAt(i)
      if (ch > 0xfff) {
        sb.append("\\u" + hex(ch))
      } else if (ch > 0xff) {
        sb.append("\\u0" + hex(ch))
      } else if (ch > 0x7f) {
        sb.append("\\u00" + hex(ch))
      } else if (ch < 32) {
        ch match {
          case '\b' =>
            sb += '\\'
            sb += 'b'
          case '\n' =>
            sb += '\\'
            sb += 'n'
          case '\t' =>
            sb += '\\'
            sb += 't'
          case '\f' =>
            sb += '\\'
            sb += 'f'
          case '\r' =>
            sb += '\\'
            sb += 'r'
          case _ =>
            if (ch > 0xf) {
              sb.append("\\u00" + hex(ch))
            }
            else {
              sb.append("\\u000" + hex(ch))
            }
        }
      } else {
        ch match {
          case '"' =>
            if (backticked)
              sb += '"'
            else {
              sb += '\\'
              sb += '\"'
            }
          case '`' =>
            if (backticked) {
              sb += '\\'
              sb += '`'
            } else
              sb += '`'
          case '\\' =>
            sb += '\\'
            sb += '\\'
          case _ =>
            sb.append(ch)
        }
      }
      i += 1
    }
    sb.result()
  }

  def unescapeString(str: String): String = unescapeString(str, new StringBuilder(capacity = str.length))

  def unescapeString(str: String, sb: StringBuilder): String = {
    sb.clear()

    val sz = str.length()
    var hadSlash = false
    var inUnicode = false
    lazy val unicode = new StringBuilder(capacity = 4)
    var i = 0
    while (i < str.length) {

      val ch = str.charAt(i)
      if (inUnicode) {
        // if in unicode, then we're reading unicode
        // values in somehow
        unicode.append(ch)
        if (unicode.length == 4) {
          // unicode now contains the four hex digits
          // which represents our unicode character
          try {
            val value = Integer.parseInt(unicode.toString(), 16)
            sb += value.toChar
            unicode.clear()
            inUnicode = false
            hadSlash = false
          } catch {
            case nfe: NumberFormatException =>
              fatal("Unable to parse unicode value: " + unicode)
          }
        }
      } else if (hadSlash) {
        hadSlash = false
        ch match {
          case '\\' => sb += '\\'
          case '\'' => sb += '\''
          case '"' => sb += '"'
          case '`' => sb += '`'
          case 'r' => sb += '\r'
          case 'f' => sb += '\f'
          case 't' => sb += '\t'
          case 'n' => sb += '\n'
          case 'b' => sb += '\b'
          case 'u' => inUnicode = true
          case _ => fatal(s"Got invalid string escape character: `\\$ch'")
        }
      } else if (ch == '\\')
        hadSlash = true
      else
        sb += ch
      i += 1
    }
    if (hadSlash) {
      // then we're in the weird case of a \ at the end of the
      // string, let's output it anyway.
      sb += '\\'
    }
    sb.result()
  }
}

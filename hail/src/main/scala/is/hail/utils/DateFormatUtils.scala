package is.hail.utils

import java.util.Locale
import java.time.format.{DateTimeFormatter, DateTimeFormatterBuilder, TextStyle}
import java.time.temporal.ChronoField
import java.time.chrono.{Chronology, ChronoLocalDate}

object DateFormatUtils {
  def parseDateFormat(str: String, locale: Locale): DateTimeFormatter = {
    val fmt = new DateTimeFormatterBuilder
    val chrono = Chronology.ofLocale(locale)
    lazy val _1970 = chrono.date(1970, 1, 1)

    def char(c: Char): Unit = fmt.appendLiteral(c)

    def spec(c: Char): Unit = c match {
      case '%' => char('%')
      case 'A' => fmt.appendText(ChronoField.DAY_OF_WEEK, TextStyle.FULL)
      case 'a' => fmt.appendText(ChronoField.DAY_OF_WEEK, TextStyle.SHORT)
      case 'B' => fmt.appendText(ChronoField.MONTH_OF_YEAR, TextStyle.FULL)
      case 'b' | 'h' => fmt.appendText(ChronoField.MONTH_OF_YEAR, TextStyle.SHORT)
      case 'D' => alternating("m/d/y")
      case 'd' => fmt.appendValue(ChronoField.DAY_OF_MONTH, 2)
      case 'e' => fmt.padNext(2); fmt.appendValue(ChronoField.DAY_OF_MONTH)
      case 'F' => alternating("Y-m-d")
      case 'G' => spec('Y')
      case 'g' => spec('y')
      case 'H' => fmt.appendValue(ChronoField.HOUR_OF_DAY, 2)
      case 'I' => fmt.appendValue(ChronoField.CLOCK_HOUR_OF_AMPM, 2)
      case 'j' => fmt.appendValue(ChronoField.DAY_OF_YEAR, 3)
      case 'k' => fmt.padNext(2); fmt.appendValue(ChronoField.HOUR_OF_DAY)
      case 'l' => fmt.padNext(2); fmt.appendValue(ChronoField.CLOCK_HOUR_OF_AMPM)
      case 'M' => fmt.appendValue(ChronoField.MINUTE_OF_HOUR, 2)
      case 'm' => fmt.appendValue(ChronoField.MONTH_OF_YEAR, 2)
      case 'n' => char('\n')
      case 'p' => fmt.appendText(ChronoField.AMPM_OF_DAY, TextStyle.SHORT)
      case 'R' => alternating("H:M")
      case 'r' => alternating("I:M:S p")
      case 'S' => fmt.appendValue(ChronoField.SECOND_OF_MINUTE, 2)
      case 's' => fmt.appendValue(ChronoField.INSTANT_SECONDS)
      case 'T' => alternating("H:M:S")
      case 't' => char('\t')
      case 'U' => fmt.appendValue(ChronoField.ALIGNED_WEEK_OF_YEAR, 2) // TODO NOTE: sunday first day
      case 'u' => fmt.appendValue(ChronoField.DAY_OF_WEEK) // NOTE: 1-7 //TODO
      case 'v' => alternating("e-b-Y")
      case 'W' => fmt.appendValue(ChronoField.ALIGNED_WEEK_OF_YEAR, 2) // TODO NOTE: monday first day
      case 'w' => fmt.appendValue(ChronoField.DAY_OF_WEEK) // NOTE: 0-6
      case 'Y' => fmt.appendValue(ChronoField.YEAR, 4)
      case 'y' => fmt.appendValueReduced(ChronoField.YEAR, 2, 2, _1970)
      case 'E' | 'O' => char(c) // Python just keeps the letter when it's unrecognized.
      case 'C' | 'c' | 'V' | 'X' | 'x' | 'Z' | 'z' => throw new HailException("Unsupported time formatting character.")
      case d => fatal(s"invalid time format descriptor: $d")
    }

    def alternating(s: String): Unit = {
      var isSpec = true
      for (c <- s) {
        if (isSpec)
          spec(c)
        else
          char(c)
        isSpec = !isSpec
      }
    }

    val chrs = str.iterator
    while (chrs.hasNext)
      chrs.next() match {
        case '%' =>
          spec(if (chrs.hasNext) chrs.next() else '%')
        case c =>
          char(c)
      }

    fmt.toFormatter
  }
}

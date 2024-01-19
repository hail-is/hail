package is.hail.utils

import java.time.DayOfWeek
import java.time.chrono.Chronology
import java.time.format.{DateTimeFormatter, DateTimeFormatterBuilder, TextStyle}
import java.time.temporal.{ChronoField, WeekFields}
import java.util.Locale

object DateFormatUtils {
  def parseDateFormat(str: String, locale: Locale): DateTimeFormatter = {
    val fmt = new DateTimeFormatterBuilder
    val chrono = Chronology.ofLocale(locale)
    lazy val _1970 = chrono.date(1970, 1, 1)

    val SUNDAY_START_ALWAYS = WeekFields.of(DayOfWeek.SUNDAY, 7)
    val MONDAY_START_ALWAYS = WeekFields.of(DayOfWeek.MONDAY, 7)

    def char(c: Char): Unit = fmt.appendLiteral(c)

    def spec(c: Char): Unit = {
      c match {
        case '%' => char('%')
        case 'A' => fmt.appendText(ChronoField.DAY_OF_WEEK, TextStyle.FULL)
        case 'a' => fmt.appendText(ChronoField.DAY_OF_WEEK, TextStyle.SHORT)
        case 'B' => fmt.appendText(ChronoField.MONTH_OF_YEAR, TextStyle.FULL)
        case 'b' | 'h' => fmt.appendText(ChronoField.MONTH_OF_YEAR, TextStyle.SHORT)
        case 'D' => alternating("m/d/y")
        case 'd' => fmt.appendValue(ChronoField.DAY_OF_MONTH, 2)
        case 'e' => fmt.padNext(2); fmt.appendValue(ChronoField.DAY_OF_MONTH)
        case 'F' => alternating("Y-m-d")
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
        case 'U' => fmt.appendValue(SUNDAY_START_ALWAYS.weekOfYear(), 2) // Sunday first day
        case 'u' => fmt.appendValue(WeekFields.ISO.dayOfWeek()) // 1-7, starts on Monday
        case 'V' => fmt.appendValue(WeekFields.ISO.weekOfWeekBasedYear(), 2)
        case 'v' => alternating("e-b-Y")
        case 'W' => fmt.appendValue(MONDAY_START_ALWAYS.weekOfYear(), 2) // Monday first day
        case 'Y' => fmt.appendValue(ChronoField.YEAR, 4)
        case 'y' => fmt.appendValueReduced(ChronoField.YEAR, 2, 2, _1970)
        case 'Z' => fmt.appendZoneId()
        case 'z' => fmt.appendOffsetId()
        case 'E' | 'O' => char(c) // Python just keeps these two letters for whatever reason.
        case 'C' | 'c' | 'G' | 'g' | 'w' | 'X' | 'x' =>
          throw new HailException(s"Currently unsupported time formatting character: $c")
        case d => fatal(s"invalid time format descriptor: $d")
      }
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

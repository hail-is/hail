package is.hail.expr

import is.hail.utils._
import is.hail.variant._

import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Position

class RichParser[T](parser: Parser.Parser[T]) {
  def parse(input: String): T =
    Parser.parseAll(parser, input) match {
      case Parser.Success(result, _) => result
      case Parser.NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

  def parseOpt(input: String): Option[T] =
    Parser.parseAll(parser, input) match {
      case Parser.Success(result, _) => Some(result)
      case Parser.NoSuccess(msg, next) => None
    }
}

object ParserUtils {
  def error(pos: Position, msg: String): Nothing = {
    val lineContents = pos.longString.split("\n").head
    val prefix = s"<input>:${pos.line}:"
    fatal(
      s"""$msg
         |$prefix$lineContents
         |${" " * prefix.length}${lineContents.take(pos.column - 1).map { c =>
          if (c == '\t') c else ' '
        }}^""".stripMargin
    )
  }

  def error(pos: Position, msg: String, tr: Truncatable): Nothing = {
    val lineContents = pos.longString.split("\n").head
    val prefix = s"<input>:${pos.line}:"
    fatal(
      s"""$msg
         |$prefix$lineContents
         |${" " * prefix.length}${lineContents.take(pos.column - 1).map { c =>
          if (c == '\t') c else ' '
        }}^""".stripMargin,
      tr,
    )
  }
}

object Parser extends JavaTokenParsers {
  def parse[T](parser: Parser[T], code: String): T =
    parseAll(parser, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

  def parseLocusInterval(input: String, rg: ReferenceGenome, invalidMissing: Boolean): Interval = {
    parseAll[Interval](locusInterval(rg, invalidMissing), input) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => fatal(
          s"""invalid interval expression: '$input': $msg
             |  Acceptable formats:
             |    CHR:POS-CHR:POS e.g. 1:12345-1:17299 or [5:151111-8:191293]
             |            An interval from the starting locus (chromosome, position)
             |            to the ending locus. By default the bounds are left-inclusive,
             |            right-exclusive, but may be configured by inclusion of square
             |            brackets ('[' or ']') for open endpoints, or parenthesis ('('
             |            or ')') for closed endpoints. The POS field may be the words
             |            'START' or 'END' to denote the start or end of the chromosome.
             |    CHR:POS-POS e.g. 1:14244-912382
             |            The same interval as '[1:14244-1:912382)'
             |    CHR-CHR e.g. '1-22' or 'X-Y'
             |            The same intervals as '[1:START-22:END'] or '[X:START-Y:END]'
             |    CHR e.g. '5' or 'X'
             |            The same intervals as '[5:START-5:END]' or '[X:START-X:END]'  """.stripMargin
        )
    }
  }

  def parseCall(input: String): Call =
    parseAll[Call](call, input) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => fatal(s"invalid call expression: '$input': $msg")
    }

  def oneOfLiteral(a: Array[String]): Parser[String] = new Parser[String] {
    private[this] val root = ParseTrieNode.generate(a)

    def apply(in: Input): ParseResult[String] = {

      var _in = in
      var node = root
      while (true) {
        if (_in.atEnd)
          if (node.value != null)
            return Success(node.value, _in)
          else
            return Failure("", in)

        val nextChar = _in.first
        val nextNode = node.search(nextChar)
        if (nextNode == null) {
          if (node.value != null)
            return Success(node.value, _in)
          else
            return Failure("", in)
        }
        _in = _in.rest
        node = nextNode
      }
      return null // unreachable
    }
  }

  def call: Parser[Call] = {
    wholeNumber ~ "/" ~ rep1sep(wholeNumber, "/") ^^ { case a0 ~ _ ~ arest =>
      CallN(coerceInt(a0) +: arest.map(coerceInt).toArray, phased = false)
    } |
      wholeNumber ~ "|" ~ rep1sep(wholeNumber, "|") ^^ { case a0 ~ _ ~ arest =>
        CallN(coerceInt(a0) +: arest.map(coerceInt).toArray, phased = true)
      } |
      wholeNumber ^^ { a => Call1(coerceInt(a), phased = false) } |
      "|" ~ wholeNumber ^^ { case _ ~ a => Call1(coerceInt(a), phased = true) } |
      "-" ^^ { _ => Call0(phased = false) } |
      "|-" ^^ { _ => Call0(phased = true) }
  }

  def intervalWithEndpoints[T](bounds: Parser[(T, T, Boolean, Boolean)]): Parser[Interval] = {
    val start = ("[" ^^^ true) | ("(" ^^^ false)
    val end = ("]" ^^^ true) | (")" ^^^ false)

    start ~ bounds ~ end ^^ { case istart ~ int ~ iend => Interval(int._1, int._2, istart, iend) } |
      bounds ^^ { int => Interval(int._1, int._2, int._3, int._4) }
  }

  def locusInterval(rgBase: ReferenceGenome, invalidMissing: Boolean): Parser[Interval] = {
    val rg = rgBase.asInstanceOf[ReferenceGenome]
    val contig = rg.contigParser

    val valueParser =
      locusUnchecked(rg) ~ "-" ~ rg.contigParser ~ ":" ~ pos ^^ { case l1 ~ _ ~ c2 ~ _ ~ p2 =>
        p2 match {
          case Some(p) => (l1, Locus(c2, p), true, false)
          case None => (l1, Locus(c2, rg.contigLength(c2)), true, true)
        }
      } |
        locusUnchecked(rg) ~ "-" ~ pos ^^ { case l1 ~ _ ~ p2 =>
          p2 match {
            case Some(p) => (l1, l1.copy(position = p), true, false)
            case None => (l1, l1.copy(position = rg.contigLength(l1.contig)), true, true)
          }
        } |
        contig ~ "-" ~ contig ^^ { case c1 ~ _ ~ c2 =>
          (Locus(c1, 1), Locus(c2, rg.contigLength(c2)), true, true)
        } |
        contig ^^ { c => (Locus(c, 1), Locus(c, rg.contigLength(c)), true, true) }
    intervalWithEndpoints(valueParser) ^^ { i => rg.toLocusInterval(i, invalidMissing) }
  }

  def locusUnchecked(rg: ReferenceGenome): Parser[Locus] =
    (rg.contigParser ~ ":" ~ pos) ^^ { case c ~ _ ~ p => Locus(c, p.getOrElse(rg.contigLength(c))) }

  def locus(rg: ReferenceGenome): Parser[Locus] =
    (rg.contigParser ~ ":" ~ pos) ^^ { case c ~ _ ~ p =>
      Locus(c, p.getOrElse(rg.contigLength(c)), rg)
    }

  def coerceInt(s: String): Int =
    try
      s.toInt
    catch {
      case e: java.lang.NumberFormatException => Int.MaxValue
    }

  def exp10(i: Int): Int = {
    var mult = 1
    var j = 0
    while (j < i) {
      mult *= 10
      j += 1
    }
    mult
  }

  def pos: Parser[Option[Int]] = {
    "[sS][Tt][Aa][Rr][Tt]".r ^^ { _ => Some(1) } |
      "[Ee][Nn][Dd]".r ^^ { _ => None } |
      "\\d+".r <~ "[Kk]".r ^^ { i => Some(coerceInt(i) * 1000) } |
      "\\d+".r <~ "[Mm]".r ^^ { i => Some(coerceInt(i) * 1000000) } |
      "\\d+".r ~ "." ~ "\\d{1,3}".r ~ "[Kk]".r ^^ { case lft ~ _ ~ rt ~ _ =>
        Some(coerceInt(lft + rt) * exp10(3 - rt.length))
      } |
      "\\d+".r ~ "." ~ "\\d{1,6}".r ~ "[Mm]".r ^^ { case lft ~ _ ~ rt ~ _ =>
        Some(coerceInt(lft + rt) * exp10(6 - rt.length))
      } |
      "\\d+".r ^^ { i => Some(coerceInt(i)) }
  }
}

package is.hail.utils.prettyPrint

import java.io.{StringWriter, Writer}
import java.util.ArrayDeque

import scala.annotation.tailrec
import scala.collection.mutable
import scala.language.implicitConversions

object PrettyPrintWriter {
  type Doc = PrettyPrintDoc

  implicit def text(t: String): Doc = Text(t)

  def space: Doc = text(" ")

  def line: Doc = Line

  def softline: Doc = group(line)

  def group(body: Doc): Doc = Group(body)

  def group(body: Iterable[Doc]): Doc = group(concat(body))

  def concat(docs: Iterable[Doc]): Doc = Concat(docs)

  def concat(docs: Doc*): Doc = Concat(docs)

  def nest(i: Int, body: Doc): Doc = Indent(i, body)

  def empty: Doc = concat(Iterable.empty)

//  def encloseSep(l: Doc, r: Doc, sep: Doc, seq: Seq[Doc]): Doc = seq match {
//    case Seq() => concat(l, r)
//    case Seq(s) => concat(l, s, r)
//    case _ =>
//      l
//      seq.head()
//      seq.tail.foreach { s => sep; s() }
//      r
//  }

  def hsep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(space))

  def hsep(docs: Doc*): Doc = hsep(docs)

  def vsep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(line))

  def vsep(docs: Doc*): Doc = vsep(docs)

  def sep(docs: Iterable[Doc]): Doc = group(vsep(docs))

  def sep(docs: Doc*): Doc = sep(docs)

  def list(docs: Iterable[Doc]): Doc =
    group(docs.intersperse(concat(text("("), line), concat(",", line), text(")")))

  def list(docs: Doc*): Doc = list(docs)

  def punctuate(punctuation: Doc, docs: Iterator[Doc]): Iterator[Doc] = new Iterator[Doc] {
    override def hasNext: Boolean = docs.hasNext

    override def next(): Doc = {
      val doc = docs.next()
      if (docs.hasNext)
        concat(doc, punctuation)
      else
        doc
    }
  }
}

object PrettyPrintDoc {
  abstract class ScanedNode
  case class TextN(t: String) extends ScanedNode
  case class LineN(indentation: Int) extends ScanedNode
  case class GroupN(contents: mutable.ArrayBuilder[ScanedNode], start: Int, var end: Int) extends ScanedNode

  abstract class KontNode
  case object PopGroup extends KontNode
  case class Unindent(indent: Int) extends KontNode
  case class Triv(kont: Iterator[PrettyPrintDoc]) extends KontNode

  def render(doc: PrettyPrintDoc, width: Int, out: Writer): Unit = {
    val buffer = new ArrayDeque[GroupN]
    val kont = new ArrayDeque[KontNode]
    var remaining: Int = width
    var globalPos: Int = 0
    var indentation: Int = 0
    var eval = doc

    def scan(d: ScanedNode, size: Int): Unit = {
      globalPos += size
      if (buffer.isEmpty) {
        printNode(d, false)
      } else {
        buffer.getLast().contents += d
        while (!buffer.isEmpty && globalPos - buffer.getFirst.start > remaining) {
          val head = buffer.removeFirst()
          head.end = globalPos
          printNode(head, false)
        }
      }
    }

    @tailrec def advance(): Unit = {
      if (kont.isEmpty) {
        eval = null
      } else {
        kont.peek() match {
          case Triv(k) =>
            if (k.isEmpty) {
              kont.pop()
              advance()
            } else {
              eval = k.next()
            }
          case PopGroup =>
            if (!buffer.isEmpty) {
              val last = buffer.removeLast()
              last.end = globalPos
              if (buffer.isEmpty) {
                printNode(last, true)
              } else {
                buffer.getLast().contents += last
              }
            }
            kont.pop()
            advance()
          case Unindent(i) =>
            indentation -= i
            kont.pop()
            advance()
        }
      }
    }

    def printNode(d: ScanedNode, horizontal: Boolean): Unit = d match {
      case TextN(t) =>
        remaining -= t.length
        out.write(t)
      case LineN(i) =>
        if (horizontal) {
          remaining -= 1
          out.write(' ')
        } else {
          out.write('\n')
          out.write(" " * i)
          remaining = width - i
        }
      case GroupN(contents, start, stop) =>
        val h = stop - start <= remaining
        contents.result().foreach(printNode(_, h))
    }

    while (eval != null) {
      eval match {
        case Text(t) =>
          scan(TextN(t), t.length)
          advance()
        case Line =>
          scan(LineN(indentation), 1)
          advance()
        case Group(body) =>
          buffer.addLast(GroupN(mutable.ArrayBuilder.make[ScanedNode], globalPos, -1))
          kont.push(PopGroup)
          eval = body
        case Indent(i, body) =>
          indentation += i
          kont.push(Unindent(i))
          eval = body
        case Concat(bodyIt) =>
          kont.push(Triv(bodyIt.iterator))
          advance()
      }
    }
  }
}

abstract class PrettyPrintDoc {
  def render(width: Int, out: Writer): Unit =
    PrettyPrintDoc.render(this, width, out)

  def render(width: Int): String = {
    val out = new StringWriter()
    render(width, out)
    out.toString
  }
}

case class Text(t: String) extends PrettyPrintDoc
case object Line extends PrettyPrintDoc
case class Group(body: PrettyPrintDoc) extends PrettyPrintDoc
case class Indent(i: Int, body: PrettyPrintDoc) extends PrettyPrintDoc
case class Concat(it: Iterable[PrettyPrintDoc]) extends PrettyPrintDoc

object OppenPPI {
  def apply(width: Int)(f: OppenPPI => Unit): String = {
    val sw = new StringWriter()
    val pp = new OppenPPI(width, sw)
    f(pp)
    sw.toString
  }
}

class OppenPPI(width: Int, out: Writer) {
  abstract class Token
  case class Text(t: String) extends Token
  case class Line(indentation: Int) extends Token
  case class Group(contents: mutable.ArrayBuilder[Token], start: Int, var end: Int) extends Token
  val dq = new ArrayDeque[Group]
  var remaining: Int = width
  var globalPos: Int = 0
  var indentation: Int = 0

  def text(t: String): Unit = scan(Text(t), t.length)

  def line(): Unit = scan(Line(indentation), 1)

  private def scan(tok: Token, size: Int): Unit = {
    globalPos += size
    if (dq.isEmpty) {
      printTok(tok, false)
    } else {
      dq.getLast().contents += tok
      while (!dq.isEmpty && globalPos - dq.getFirst.start > remaining) {
        val head = dq.removeFirst()
        head.end = globalPos
        printTok(head, false)
      }
    }
  }

  def group(d: => Unit): Unit = {
    dq.addLast(Group(mutable.ArrayBuilder.make[Token], globalPos, -1))
    d
    if (!dq.isEmpty) {
      val last = dq.removeLast()
      last.end = globalPos
      if (dq.isEmpty) {
        printTok(last, true)
      } else {
        dq.getLast().contents += last
      }
    }
  }

  def nest(j: Int)(d: => Unit): Unit = {
    indentation += j
    d
    indentation -= j
  }

  private def printTok(t: Token, horizontal: Boolean): Unit = t match {
    case Text(t) =>
      remaining -= t.length
      out.write(t)
    case Line(i) =>
      if (horizontal) {
        remaining -= 1
        out.write(' ')
      } else {
        out.write('\n')
        out.write(" " * i)
        remaining = width - i
      }
    case Group(contents, start, stop) =>
      val h = stop - start <= remaining
      contents.result().foreach(printTok(_, h))
  }
}

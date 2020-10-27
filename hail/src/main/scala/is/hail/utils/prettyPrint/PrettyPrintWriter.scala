package is.hail.utils.prettyPrint

import java.io.{StringWriter, Writer}
import java.util.ArrayDeque

import scala.annotation.tailrec
import scala.collection.mutable

object Doc {
  def render(doc: Doc, width: Int, out: Writer): Unit = {
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
      case LineN(i, ifFlat: String) =>
        if (horizontal) {
          remaining -= ifFlat.length
          out.write(ifFlat)
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
        case Line(ifFlat) =>
          scan(LineN(indentation, ifFlat), ifFlat.length)
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

abstract class Doc {
  def render(width: Int, out: Writer): Unit =
    Doc.render(this, width, out)

  def render(width: Int): String = {
    val out = new StringWriter()
    render(width, out)
    out.toString
  }
}

private[prettyPrint] case class Text(t: String) extends Doc
private[prettyPrint] case class Line(ifFlat: String) extends Doc
private[prettyPrint] case class Group(body: Doc) extends Doc
private[prettyPrint] case class Indent(i: Int, body: Doc) extends Doc
private[prettyPrint] case class Concat(it: Iterable[Doc]) extends Doc

private[prettyPrint] abstract class ScanedNode
private[prettyPrint] case class TextN(t: String) extends ScanedNode
private[prettyPrint] case class LineN(indentation: Int, ifFlat: String) extends ScanedNode
private[prettyPrint] case class GroupN(contents: mutable.ArrayBuilder[ScanedNode], start: Int, var end: Int) extends ScanedNode

private[prettyPrint] abstract class KontNode
private[prettyPrint] case object PopGroup extends KontNode
private[prettyPrint] case class Unindent(indent: Int) extends KontNode
private[prettyPrint] case class Triv(kont: Iterator[Doc]) extends KontNode


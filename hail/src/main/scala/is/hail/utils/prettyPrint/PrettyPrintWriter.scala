package is.hail.utils.prettyPrint

import java.io.{StringWriter, Writer}
import java.util.ArrayDeque

import is.hail.utils.ArrayBuilder

import scala.annotation.tailrec

object Doc {
  def render(doc: Doc, width: Int, ribbonWidth: Int, _maxLines: Int, out: Writer): Unit = {
    val buffer = new ArrayDeque[GroupN]
    val kont = new ArrayDeque[KontNode]
    val maxLines = if (_maxLines > 0) _maxLines else Int.MaxValue
    var lines: Int = 0
    var remaining: Int = math.min(width, ribbonWidth)
    var globalPos: Int = 0
    var indentation: Int = 0
    var eval = doc

    var pendingPushes: Int = 0
    var pendingPops: Int = 0

    def scan(d: ScanedNode, size: Int): Unit = {
      globalPos += size
      if (buffer.isEmpty) {
        printNode(d, false)
      } else {
        buffer.getLast.contents += d
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
            if (pendingPushes > 0) pendingPushes -= 1 else pendingPops += 1
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
          lines += 1
          if (lines >= maxLines) throw new MaxLinesExceeded()
          out.write('\n')
          out.write(" " * i)
          remaining = math.min(width - i, ribbonWidth)
        }
      case GroupN(contents, start, stop) =>
        val h = stop - start <= remaining
        var i = 0
        while (i < contents.size) {
          printNode(contents(i), h)
          i += 1
        }
    }

    def closeGroups(): Unit =
      while (pendingPops > 0) {
        if (buffer.isEmpty) {
          pendingPops = 0
          return
        }
        val last = buffer.removeLast()
        last.end = globalPos
        if (buffer.isEmpty) {
          printNode(last, true)
        } else {
          buffer.getLast.contents += last
        }
        pendingPops -= 1
      }

    def openGroups(): Unit = {
      while (pendingPushes > 0) {
        buffer.addLast(GroupN(new ArrayBuilder[ScanedNode](), globalPos, -1))
        pendingPushes -= 1
      }
    }

    try {
      while (eval != null) {
        eval match {
          case Text(t) =>
            scan(TextN(t), t.length)
            advance()
          case Line(ifFlat) =>
            closeGroups()
            openGroups()
            scan(LineN(indentation, ifFlat), ifFlat.length)
            advance()
          case Group(body) =>
            kont.push(PopGroup)
            pendingPushes += 1
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
      closeGroups()
    } catch {
      case _: MaxLinesExceeded =>
    }
  }
}

abstract class Doc {
  def render(width: Int, ribbonWidth: Int, maxLines: Int, out: Writer): Unit =
    Doc.render(this, width, ribbonWidth, maxLines, out)

  def render(width: Int, ribbonWidth: Int, maxLines: Int): String = {
    val out = new StringWriter()
    render(width, ribbonWidth, maxLines, out)
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
private[prettyPrint] case class GroupN(contents: ArrayBuilder[ScanedNode], start: Int, var end: Int) extends ScanedNode

private[prettyPrint] abstract class KontNode
private[prettyPrint] case object PopGroup extends KontNode
private[prettyPrint] case class Unindent(indent: Int) extends KontNode
private[prettyPrint] case class Triv(kont: Iterator[Doc]) extends KontNode

class MaxLinesExceeded() extends Exception
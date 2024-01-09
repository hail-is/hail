package is.hail.utils.prettyPrint

import is.hail.utils.BoxedArrayBuilder

import java.io.{StringWriter, Writer}
import java.util.ArrayDeque
import scala.annotation.tailrec

object Doc {
  def render(doc: Doc, width: Int, ribbonWidth: Int, _maxLines: Int, out: Writer): Unit = {
    // All groups whose formatting is still undetermined. The innermost group is at the end.
    // A group which has been determined to be non-flat is popped from the front, even if the
    // group is still open. This is safe because top-level Lines (not in any group) always print
    // as newlines.
    // Each GroupN contains the contents of the group that has been scanned so far.
    val pendingGroups = new ArrayDeque[GroupN]

    // Represents the rest of the document past the node currently being scanned.
    val kont = new ArrayDeque[KontNode]

    // The current position in the document, if it had been formatted entirely in one line.
    // This is only used to take the difference between two globalPos values, to determine
    // the length of a group if it is formatted flat.
    var globalPos: Int = 0

    val maxLines = if (_maxLines > 0) _maxLines else Int.MaxValue
    var lines: Int = 0
    var remainingInLine: Int = math.min(width, ribbonWidth)
    var indentation: Int = 0
    var currentNode = doc

    // Group openings and closes are deferred until the next Line. This forces any Text to be
    // considered part of the group containing the previous Line. For opens, this is a slight
    // performance optimization, because text at the beginning of a group can be written eagerly,
    // regardless of how the group is formatted. For closes, this is more correct: if text
    // immediately follows the end of a group, then the group may fit on the current line, yet
    // the following text would exceed the max width.
    var pendingOpens: Int = 0
    var pendingCloses: Int = 0

    def scan(node: ScannedNode, size: Int): Unit = {
      globalPos += size
      if (pendingGroups.isEmpty) {
        printNode(node, false)
      } else {
        pendingGroups.getLast.contents += node
        while (
          !pendingGroups.isEmpty && globalPos - pendingGroups.getFirst.start > remainingInLine
        ) {
          val head = pendingGroups.removeFirst()
          head.end = globalPos
          printNode(head, false)
        }
      }
    }

    // Process the top of kont until a non-empty ConcatK is found; move the first contained node
    // to currentNode.
    @tailrec def advance(): Unit = {
      if (kont.isEmpty) {
        currentNode = null
      } else {
        kont.peek() match {
          case ConcatK(k) =>
            if (k.isEmpty) {
              kont.pop()
              advance()
            } else {
              currentNode = k.next()
            }
          case PopGroupK =>
            if (pendingOpens > 0) pendingOpens -= 1 else pendingCloses += 1
            kont.pop()
            advance()
          case UnindentK(i) =>
            indentation -= i
            kont.pop()
            advance()
        }
      }
    }

    def printNode(node: ScannedNode, flatten: Boolean): Unit = node match {
      case TextN(t) =>
        remainingInLine -= t.length
        out.write(t)
      case LineN(i, ifFlat: String) =>
        if (flatten) {
          remainingInLine -= ifFlat.length
          out.write(ifFlat)
        } else {
          lines += 1
          if (lines >= maxLines) throw new MaxLinesExceeded()
          out.write('\n')
          out.write(" " * i)
          remainingInLine = math.min(width - i, ribbonWidth)
        }
      case GroupN(contents, start, stop) =>
        val h = stop - start <= remainingInLine
        var i = 0
        while (i < contents.size) {
          printNode(contents(i), h)
          i += 1
        }
    }

    def closeGroups(): Unit =
      while (pendingCloses > 0) {
        if (pendingGroups.isEmpty) {
          pendingCloses = 0
          return
        }
        val last = pendingGroups.removeLast()
        last.end = globalPos
        if (pendingGroups.isEmpty) {
          printNode(last, true)
        } else {
          pendingGroups.getLast.contents += last
        }
        pendingCloses -= 1
      }

    def openGroups(): Unit =
      while (pendingOpens > 0) {
        pendingGroups.addLast(GroupN(new BoxedArrayBuilder[ScannedNode](), globalPos, -1))
        pendingOpens -= 1
      }

    try {
      while (currentNode != null) {
        currentNode match {
          case Text(t) =>
            scan(TextN(t), t.length)
            advance()
          case Line(ifFlat) =>
            closeGroups()
            openGroups()
            scan(LineN(indentation, ifFlat), ifFlat.length)
            advance()
          case Group(body) =>
            kont.push(PopGroupK)
            pendingOpens += 1
            currentNode = body
          case Indent(i, body) =>
            indentation += i
            kont.push(UnindentK(i))
            currentNode = body
          case Concat(bodyIt) =>
            kont.push(ConcatK(bodyIt.iterator))
            advance()
        }
      }
      closeGroups()
    } catch {
      case _: MaxLinesExceeded =>
      // 'maxLines' have been printed, so break out of the loop and stop printing.
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

abstract private[prettyPrint] class ScannedNode
private[prettyPrint] case class TextN(t: String) extends ScannedNode
private[prettyPrint] case class LineN(indentation: Int, ifFlat: String) extends ScannedNode

private[prettyPrint] case class GroupN(
  contents: BoxedArrayBuilder[ScannedNode],
  start: Int,
  var end: Int,
) extends ScannedNode

abstract private[prettyPrint] class KontNode
private[prettyPrint] case object PopGroupK extends KontNode
private[prettyPrint] case class UnindentK(indent: Int) extends KontNode
private[prettyPrint] case class ConcatK(kont: Iterator[Doc]) extends KontNode

private[prettyPrint] class MaxLinesExceeded() extends Exception

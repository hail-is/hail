package is.hail.lir

import java.io.{StringWriter, Writer}

import scala.collection.mutable

class PrettyPrintWriter(var n: Int, out: Writer, val printSourceLineNumbers: Boolean = false) {
  var lineNumber: Int = 0

  def indent(f: => Unit): Unit = {
    n += 2
    f
    n -= 2
  }

  def +=(s: String): Unit = appendWithSource(s, 0)

  def appendWithSource(s: String, sourceNum: Int): Unit = {
    lineNumber += 1
    if (lineNumber > 1) out.write('\n')
    out.write(f"$lineNumber%-4d ")
    if (printSourceLineNumbers) {
      if (sourceNum == 0) out.write("     ") else out.write(f"$sourceNum%-4d ")
    }
    out.write(" " * n)
    out.write(s)
  }

  def appendToLastLine(s: String): Unit = {
    out.write(s)
  }
}

object OppenPP {
  type Remaining = Int
  type Position = Int
  type Indent = Int
  type Width = Int
  type Horizontal = Boolean
  type Out = Remaining => String
  type OutGroup = (Horizontal, Out) => Out
  type Dq = Vector[(Position, OutGroup)]
  type TreeCont = (Position, Dq) => Out
  type Cont = ((Indent, Width), TreeCont) => TreeCont

  def text(t: String): Cont = (iw, k) => {
    val l = t.length
    scan(l, (_, c) => r => t ++ c(r - l), k)
  }

  def line(): Cont = { case ((i, w), k) =>
    scan(1, (h, c) => r => if (h) s" ${c(r - 1)}" else s"\n${" " * i}${c(w - i)}", k)
  }

  def group(d: Cont): Cont = (iw, c) => (p, dq) => {
    d(iw, leave(c))(p, dq :+ ((p, (h: Horizontal, c: Out) => c)))
  }

  def concat(dl: Cont, dr: Cont): Cont = (iw, c) => dl(iw, dr(iw, c))

  def nest(j: Int, d: Cont): Cont = { case ((i, w), c) => d((i + j, w), c) }

  def pretty(width: Int, d: Cont): String =
    d((0, width), (p, dq) => r => "")(0, Vector.empty)(width)

  private def scan(l: Width, out: OutGroup, c: TreeCont): TreeCont = (p, dq) => {
    if (dq.isEmpty)
      out(false, c(p + l, dq))
    else {
      val (s, grp) = dq.last
      prune(c)(p + l, dq.init :+ ((s, (h: Horizontal, o: Out) => grp(h, out(h, o)))))
    }
  }

  private def prune(c: TreeCont): TreeCont = (p, dq) => r => {
    if (dq.isEmpty)
      c(p, dq)(r)
    else {
      val (s, grp) = dq.head
      if (p > s + r)
        grp(false, prune(c)(p, dq.tail))(r)
      else
        c(p, dq)(r)
    }
  }

  private def leave(c: TreeCont): TreeCont = (p, dq) => {
    if (dq.isEmpty)
      c(p, dq)
    else {
      val (s1, grp1) = dq.last
      val dq1 = dq.init
      if (dq1.isEmpty)
        grp1(true, c(p, dq1))
      else {
        val (s2, grp2) = dq1.last
        val pp = dq1.init
        c(p, pp :+ ((s2, (h: Horizontal, c: Out) => grp2(h, r => grp1(p <= s1 + r, c)(r)))))
      }
    }
  }
}

class OppenPP2(width: Int) {
  type Remaining = Int
  type Position = Int
  type Indent = Int
  type Width = Int
  type Horizontal = Boolean
  type Out = Remaining => String
  type OutGroup = (Horizontal, Out) => Out
  type Dq = Vector[(Position, OutGroup)]
  type TreeCont = (Position, Dq) => Out
  type Cont = (Indent, TreeCont) => TreeCont

  def text(t: String): Cont = (i: Indent, k: TreeCont) => {
    val l = t.length
    scan(l, (_: Horizontal, c: Out) => (r: Remaining) => t ++ c(r - l), k)
  }

  def line(): Cont = (i: Indent, k: TreeCont) =>
    scan(1, (h: Horizontal, c: Out) => (r: Remaining) => if (h) s" ${c(r - 1)}" else s"\n${" " * i}${c(width - i)}", k)

  def group(d: Cont): Cont = (i: Indent, k: TreeCont) => (p: Position, dq: Dq) => {
    d(i, leave(k))(p, dq :+ ((p, (h: Horizontal, c: Out) => c)))
  }

  def concat(dl: Cont, dr: Cont): Cont = (i: Indent, k: TreeCont) =>
    dl(i, dr(i, k))

  def nest(j: Int, d: Cont): Cont = { case (i, c) => d(i + j, c) }

  def pretty(d: Cont): String =
    d(0, (p, dq) => r => "")(0, Vector.empty)(width)

  private def scan(l: Width, out: OutGroup, c: TreeCont): TreeCont = (p: Position, dq: Dq) => {
    if (dq.isEmpty)
      out(false, c(p + l, dq))
    else {
      val (s: Position, grp: OutGroup) = dq.last
      prune(c)(p + l, dq.init :+ ((s, (h: Horizontal, o: Out) => grp(h, out(h, o)))))
    }
  }

  private def prune(c: TreeCont): TreeCont = (p: Position, dq: Dq) => r => {
    if (dq.isEmpty)
      c(p, dq)(r)
    else {
      val (s, grp) = dq.head
      if (p > s + r)
        grp(false, prune(c)(p, dq.tail))(r)
      else
        c(p, dq)(r)
    }
  }

  private def leave(c: TreeCont): TreeCont = (p: Position, dq: Dq) => {
    if (dq.isEmpty)
      c(p, dq)
    else {
      val (s1, grp1) = dq.last
      val dq1 = dq.init
      if (dq1.isEmpty)
        grp1(true, c(p, dq1))
      else {
        val (s2, grp2) = dq1.last
        val pp = dq1.init
        c(p, pp :+ ((s2, (h: Horizontal, c: Out) => grp2(h, r => grp1(p <= s1 + r, c)(r)))))
      }
    }
  }
}

// defunctionalize OutGroup
class OppenPP3(width: Int) {
  type Remaining = Int
  type Position = Int
  type Indent = Int
  type Width = Int
  type Horizontal = Boolean

  abstract class Group
  case class GroupText(t: String) extends Group
  case class GroupLine(i: Indent) extends Group
  case object GroupEmpty extends Group
  case class GroupConcat(l: Group, r: Group) extends Group
  case class GroupNest(grp1: Group, grp2: Group, p: Position, s1: Position) extends Group

  // buffer of groups, with starting positions
  type Dq = Vector[(Position, Group)]

  type Out = Remaining => String
  type TreeCont = (Position, Dq) => Out
  type Cont = (Indent, TreeCont) => TreeCont

  def text(t: String): Cont = (i: Indent, k: TreeCont) => {
    val l = t.length
    val grp = GroupText(t)
    scan(l, grp, k)
  }

  def line(): Cont = (i: Indent, k: TreeCont) => {
    val grp = GroupLine(i)
    scan(1, grp, k)
  }

  def group(d: Cont): Cont = (i: Indent, k: TreeCont) => (p: Position, dq: Dq) => {
    d(i, leave(k))(p, dq :+ ((p, GroupEmpty)))
  }

  def concat(dl: Cont, dr: Cont): Cont = (i: Indent, k: TreeCont) =>
    dl(i, dr(i, k))

  def nest(j: Int, d: Cont): Cont = { case (i, c) => d(i + j, c) }

  def pretty(d: Cont): String =
    d(0, (p, dq) => r => "")(0, Vector.empty)(width)

  private def printGrp(grp: Group, h: Horizontal, c: Out): Out = grp match {
    case GroupText(t) =>
      r => t ++ c(r - t.length)
    case GroupLine(i) =>
      r => if (h) s" ${c(r - 1)}" else s"\n${" " * i}${c(width - i)}"
    case GroupEmpty => c
    case GroupConcat(l, r) =>
      printGrp(l, h, printGrp(r, h, c))
    case GroupNest(grp1, grp2, p, s1) =>
      printGrp(grp2, h, r => printGrp(grp1, p <= s1 + r, c)(r))
  }

  private def scan(l: Width, newGrp: Group, k: TreeCont): TreeCont = (p: Position, dq: Dq) => {
    if (dq.isEmpty) {
      // not inside any undecided groups
      printGrp(newGrp, false, k(p + l, dq))
    } else {
      val (s: Position, innerGrp: Group) = dq.last
      // TODO: replace prune by loop
      prune(k)(p + l, dq.init :+ ((s, GroupConcat(innerGrp, newGrp))))
    }
  }

  private def prune(k: TreeCont): TreeCont = (p: Position, dq: Dq) => r => {
    if (dq.isEmpty)
      k(p, dq)(r)
    else {
      val (s, grp) = dq.head
      if (p > s + r)
        printGrp(grp, false, prune(k)(p, dq.tail))(r)
      else
        k(p, dq)(r)
    }
  }

  private def leave(k: TreeCont): TreeCont = (p: Position, dq: Dq) => {
    if (dq.isEmpty)
      k(p, dq)
    else {
      val (s1, grp1) = dq.last
      val dq1 = dq.init
      if (dq1.isEmpty)
        printGrp(grp1, true, k(p, dq1))
      else {
        val (s2, grp2) = dq1.last
        val pp = dq1.init
        val newGrp = GroupNest(grp1, grp2, p, s1)
        k(p, pp :+ ((s2, newGrp)))
      }
    }
  }
}

// Mutable Remaining
class OppenPP4(width: Int) {
  type Remaining = Int
  type Position = Int
  type Indent = Int
  type Width = Int
  type Horizontal = Boolean

  abstract class Group
  case class GroupText(t: String) extends Group
  case class GroupLine(i: Indent) extends Group
  case object GroupEmpty extends Group
  case class GroupConcat(l: Group, r: Group) extends Group
  case class GroupNest(grp1: Group, grp2: Group, p: Position, s1: Position) extends Group

  // buffer of groups, with starting positions
  type Dq = Vector[(Position, Group)]

  // Out => Out is type of computations which print to out stream and mutate Remaining
  type Out = () => String
  type TreeCont = (Position, Dq) => Out
  type Cont = (Indent, TreeCont) => TreeCont

  var r: Remaining = width

  def text(t: String): Cont = (i: Indent, k: TreeCont) => {
    val l = t.length
    val grp = GroupText(t)
    scan(l, grp, k)
  }

  def line(): Cont = (i: Indent, k: TreeCont) => {
    val grp = GroupLine(i)
    scan(1, grp, k)
  }

  def group(d: Cont): Cont = (i: Indent, k: TreeCont) => (p: Position, dq: Dq) => {
    d(i, leave(k))(p, dq :+ ((p, GroupEmpty)))
  }

  def concat(dl: Cont, dr: Cont): Cont = (i: Indent, k: TreeCont) =>
    dl(i, dr(i, k))

  def nest(j: Int, d: Cont): Cont = { case (i, c) => d(i + j, c) }

  def pretty(d: Cont): String =
    d(0, (p, dq) => () => "")(0, Vector.empty)()

  private def printGrp(grp: Group, h: Horizontal, c: Out): Out = grp match {
    case GroupText(t) => () => {
      r = r - t.length
      t ++ c()
    }
    case GroupLine(i) => () =>
      if (h) {
        r = r - 1
        s" ${c()}"
      } else {
        r = width - i
        s"\n${" " * i}${c()}"
      }
    case GroupEmpty => c
    case GroupConcat(l, r) =>
      printGrp(l, h, printGrp(r, h, c))
    case GroupNest(grp1, grp2, p, s1) =>
      printGrp(grp2, h, () => printGrp(grp1, p <= s1 + r, c)())
  }

  private def scan(l: Width, newGrp: Group, k: TreeCont): TreeCont = (p: Position, dq: Dq) => {
    if (dq.isEmpty) {
      // not inside any undecided groups
      printGrp(newGrp, false, k(p + l, dq))
    } else {
      val (s: Position, innerGrp: Group) = dq.last
      // TODO: replace prune by loop
      prune(k)(p + l, dq.init :+ ((s, GroupConcat(innerGrp, newGrp))))
    }
  }

  private def prune(k: TreeCont): TreeCont = (p: Position, dq: Dq) => () => {
    if (dq.isEmpty)
      k(p, dq)()
    else {
      val (s, grp) = dq.head
      if (p > s + r)
        printGrp(grp, false, prune(k)(p, dq.tail))()
      else
        k(p, dq)()
    }
  }

  private def leave(k: TreeCont): TreeCont = (p: Position, dq: Dq) => {
    if (dq.isEmpty)
      k(p, dq)
    else {
      val (s1, grp1) = dq.last
      val dq1 = dq.init
      if (dq1.isEmpty)
        printGrp(grp1, true, k(p, dq1))
      else {
        val (s2, grp2) = dq1.last
        val pp = dq1.init
        val newGrp = GroupNest(grp1, grp2, p, s1)
        k(p, pp :+ ((s2, newGrp)))
      }
    }
  }
}

// Mutable StringBuilder, no Out continuations
class OppenPP5(width: Int) {
  type Remaining = Int
  type Position = Int
  type Indent = Int
  type Width = Int
  type Horizontal = Boolean

  abstract class Group
  case class GroupText(t: String) extends Group
  case class GroupLine(i: Indent) extends Group
  case object GroupEmpty extends Group
  case class GroupConcat(l: Group, r: Group) extends Group
  case class GroupNest(grp1: Group, grp2: Group, p: Position, s1: Position) extends Group

  // buffer of groups, with starting positions
  type Dq = Vector[(Position, Group)]

  // Out => Out is type of computations which print to out stream and mutate Remaining
  type Out = () => String
  type TreeCont = (Position, Dq) => Out
  // TreeCont => TreeCont is type of computations which print to out stream and mutate Remaining, Position, and Dq
  type Cont = (Indent, TreeCont) => TreeCont

  var r: Remaining = width
  var p: Position = 0
  var dq: Dq = Vector()
  val out = new StringBuilder()

  def text(t: String): Indent => Unit = i => {
    val l = t.length
    val grp = GroupText(t)
    scan(l, grp)
  }

  def line(): Indent => Unit = i => {
    val grp = GroupLine(i)
    scan(1, grp)
  }

  def group(d: Indent => Unit): Indent => Unit = i => {
    dq = dq :+ ((p, GroupEmpty))
    d(i)
    leave()
  }

  def concat(dl: Indent => Unit, dr: Indent => Unit): Indent => Unit = i => {
    dl(i)
    dr(i)
  }

  def nest(j: Int, d: Indent => Unit): Indent => Unit = i => d(i + j)

  def pretty(d: Indent => Unit): String = {
    d(0)
    out.result()
  }

  private def printGrp(grp: Group, h: Horizontal): Unit = grp match {
    case GroupText(t) =>
      r = r - t.length
      out ++= t
    case GroupLine(i) =>
      if (h) {
        r = r - 1
        out += ' '
      } else {
        r = width - i
        out += '\n'
        out ++= " " * i
      }
    case GroupEmpty =>
    case GroupConcat(l, r) =>
      printGrp(l, h)
      printGrp(r, h)
    case GroupNest(grp1, grp2, p, s1) =>
      printGrp(grp2, h)
      printGrp(grp1, p - s1 <= r)
  }

  private def scan(l: Width, newGrp: Group): Unit = {
    p = p + l
    if (dq.isEmpty) {
      // not inside any undecided groups
      printGrp(newGrp, false)
    } else {
      val (s: Position, innerGrp: Group) = dq.last
      // TODO: replace prune by loop
      dq = dq.init :+ ((s, GroupConcat(innerGrp, newGrp)))
      prune()
    }
  }

  private def prune(): Unit = {
    if (dq.nonEmpty) {
      val (s, grp) = dq.head
      if (p > s + r) {
        printGrp(grp, false)
        dq = dq.tail
        prune()
      }
    }
  }

  private def leave(): Unit = {
    if (dq.nonEmpty) {
      val (s1, grp1) = dq.last
      val dq1 = dq.init
      if (dq1.isEmpty) {
        printGrp(grp1, true)
        dq = dq1
      } else {
        val (s2, grp2) = dq1.last
        val pp = dq1.init
        val newGrp = GroupNest(grp1, grp2, p, s1)
        dq = pp :+ ((s2, newGrp))
      }
    }
  }
}

object OppenPPI {
  def apply(width: Int)(f: OppenPPI => Unit): String = {
    val sw = new StringWriter()
    val pp = new OppenPPI(width, sw)
    f(pp)
    sw.toString
  }
}

class OppenPPI(width: Int, out: Writer) {
  type Width = Int
  abstract class Token
  case class Text(t: String) extends Token
  case class Line(indentation: Int) extends Token
  case class Group(contents: mutable.ArrayBuilder[Token], start: Int, var end: Int) extends Token
  val dq = new java.util.ArrayDeque[Group]
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

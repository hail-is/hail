package is.hail.lir

import org.objectweb.asm

class Builder(var n: Int) {
  val sb: StringBuilder = new StringBuilder()

  private var indented: Boolean = false

  def indent(f: => Unit): Unit = {
    n += 2
    f
    n -= 2
  }

  def +=(s: String): Unit = {
    if (!indented) {
      sb.append(" " * n)
      indented = true
    }

    var i = 0
    while (i < s.length) {
      val c = s(i)
      sb += c
      if (c == '\n')
        sb.append(" " * n)
      i += 1
    }
  }

  def result(): String = sb.result()
}

object Pretty {
  def apply(c: Classx[_]): String = {
    val b = new Builder(0)
    fmt(c, b)
    b.result()
  }

  def apply(m: Method): String = {
    val b = new Builder(0)
    fmt(m, b)
    b.result()
  }

  def apply(x: X): String = {
    val b = new Builder(0)
    fmt(x, b)
    b.result()
  }

  def fmt(c: Classx[_], b: Builder): Unit = {
    // FIXME interfaces
    b += s"class ${ c.name } extends ${ c.superName }"

    b.indent {
      b += "\n"
      for (f <- c.fields) {
        b += s"field ${ f.name } ${ f.ti.desc }\n"
      }
    }

    b.indent {
      b += "\n"
      for (m <- c.methods) {
        fmt(m, b)
      }
    }
  }

  def fmt(m: Method, b: Builder): Unit = {
    val blocks = m.findBlocks()

    b += s"def ${ m.name } (${ m.parameterTypeInfo.map(_.desc).mkString(",") })${ m.returnTypeInfo.desc }"

    b.indent {
      b += "\n"
      b += s"entry ${ m.entry }\n"
      for (ell <- blocks) {
        fmt(ell, b)
      }
    }

    b += "\n"
  }

  def fmt(L: Block, b: Builder): Unit = {
    b += s"$L:"

    b.indent {
      var x = L.first
      while (x != null) {
        b += "\n"
        fmt(x, b)
        x = x.next
      }
    }

    b += "\n"
  }

  def fmt(x: X, b: Builder): Unit = {
    val cl = x.getClass.getSimpleName
    val h = header(x)
    if (h != "")
      b += s"($cl $h"
    else
      b += s"($cl"
    b.indent {
      for (c <- x.children) {
        b += "\n"
        if (c != null)
          fmt(c, b)
        else
          b += "null"
      }
    }
    b += ")"
  }

  def header(x: X): String = x match {
    case x: IfX => s"${ asm.util.Printer.OPCODES(x.op) } ${ x.Ltrue } ${ x.Lfalse }"
    case x: GotoX => x.L.toString
    case x: SwitchX => s"${ x.Ldefault } (${ x.Lcases.mkString(" ") })"
    case x: LdcX => x.a.toString
    case x: InsnX => asm.util.Printer.OPCODES(x.op)
    case x: StoreX => x.l.toString
    case x: IincX => s"${ x.l.toString } ${ x.i }"
    case x: PutFieldX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.f }"
    case x: GetFieldX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.f }"
    case x: NewInstanceX => x.ti.iname
    case x: TypeInsnX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.t }"
    case x: NewArrayX => x.eti.desc
    case x: MethodX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.method }"
    case x: MethodStmtX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.method }"
    case x: LoadX => x.l.toString
    case x: StmtOpX => asm.util.Printer.OPCODES(x.op)
    case _ =>
      ""
  }
}

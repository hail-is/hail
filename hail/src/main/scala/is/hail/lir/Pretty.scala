package is.hail.lir

import org.objectweb.asm

class Builder(var n: Int) {
  var lineNumber: Int = 0

  private val sb: StringBuilder = new StringBuilder()

  def indent(f: => Unit): Unit = {
    n += 2
    f
    n -= 2
  }

  def +=(s: String): Unit = {
    if (lineNumber > 0)
      sb.append('\n')
    sb.append(f"$lineNumber%-4d ")
    lineNumber += 1
    sb.append(" " * n)
    sb.append(s)
  }

  def appendToLastLine(s: String): Unit = {
    sb.append(s)
  }

  def result(): String = {
    sb.append('\n')
    sb.result()
  }
}

object Pretty {
  def apply(c: Classx[_], saveLineNumbers: Boolean = false): String = {
    val b = new Builder(0)
    fmt(c, b, saveLineNumbers)
    b.result()
  }

  def apply(m: Method): String = {
    val b = new Builder(0)
    fmt(m, b, false)
    b.result()
  }

  def apply(L: Block): String = {
    val b = new Builder(0)
    val label: Block => String = _.toString
    fmt(L, label, b, false)
    b.result()
  }

  def apply(x: X): String = {
    val b = new Builder(0)
    val label: Block => String = _.toString
    fmt(x, label, b)
    b.result()
  }

  def fmt(c: Classx[_], b: Builder, saveLineNumbers: Boolean): Unit = {
    // FIXME interfaces
    b += s"class ${ c.name } extends ${ c.superName }"

    b.indent {
      for (f <- c.fields) {
        b += s"field ${ f.name } ${ f.ti.desc }"
      }
      b += ""
      for (m <- c.methods) {
        fmt(m, b, saveLineNumbers)
      }
    }
  }

  def fmt(m: Method, b: Builder, saveLineNumbers: Boolean): Unit = {
    val blocks = m.findBlocks()

    b += s"def ${ m.name } (${ m.parameterTypeInfo.map(_.desc).mkString(",") })${ m.returnTypeInfo.desc }"

    val label: Block => String = b => s"L${ blocks.index(b) }"

    b.indent {
      b += s"entry L${ blocks.index(m.entry) }"
      for (ell <- blocks) {
        fmt(ell, label, b, saveLineNumbers)
      }
    }

    b += ""
  }

  def fmt(L: Block, label: Block => String, b: Builder, saveLineNumbers: Boolean): Unit = {
    b += s"${ label(L) }:"

    b.indent {
      var x = L.first
      while (x != null) {
        assert(saveLineNumbers)
        if (saveLineNumbers)
          x.lineNumber = b.lineNumber
        fmt(x, label, b)
        x = x.next
      }
    }
  }

  def fmt(x: X, label: Block => String, b: Builder): Unit = {
    val cl = x.getClass.getSimpleName
    val h = header(x, label)
    if (h != "")
      b += s"($cl $h"
    else
      b += s"($cl"
    b.indent {
      for (c <- x.children) {
        if (c != null)
          fmt(c, label, b)
        else
          b += "null"
      }
      b.appendToLastLine(")")
    }
  }

  def header(x: X, label: Block => String): String = x match {
    case x: IfX => s"${ asm.util.Printer.OPCODES(x.op) } ${ label(x.Ltrue) } ${ label(x.Lfalse) }"
    case x: GotoX =>
      if (x.L != null)
        label(x.L)
      else
        "null"
    case x: SwitchX => s"${ label(x.Ldefault) } (${ x.Lcases.map(label).mkString(" ") })"
    case x: LdcX => s"${ x.a.toString } ${ x.ti }"
    case x: InsnX => asm.util.Printer.OPCODES(x.op)
    case x: StoreX => x.l.toString
    case x: IincX => s"${ x.l.toString } ${ x.i }"
    case x: PutFieldX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.f }"
    case x: GetFieldX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.f }"
    case x: NewInstanceX => s"${ x.ti.iname } ${ x.ctor }"
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

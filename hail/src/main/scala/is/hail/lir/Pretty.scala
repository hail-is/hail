package is.hail.lir

import java.io.{StringWriter, Writer}

import org.objectweb.asm

import is.hail.utils.StringEscapeUtils.escapeString

class Builder(var n: Int, out: Writer, val printSourceLineNumbers: Boolean = false) {
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

object Pretty {
  def apply(c: Classx[_]): String = apply(c, false)

  def apply(c: Classx[_], saveLineNumbers: Boolean): String = {
    val sw = new StringWriter()
    apply(c, sw, saveLineNumbers)
    sw.toString
  }

  def apply(c: Classx[_], out: Writer, saveLineNumbers: Boolean): Unit = {
    val printSourceLineNumbers = c.sourceFile.nonEmpty && saveLineNumbers
    val b = new Builder(0, out, printSourceLineNumbers)
    fmt(c, b, saveLineNumbers)
    b += ""
  }

  def apply(m: Method): String = {
    val sw = new StringWriter()
    val b = new Builder(0, sw)
    fmt(m, b, false)
    b += ""
    sw.toString
  }

  def apply(L: Block): String = {
    val sw = new StringWriter()
    val b = new Builder(0, sw)
    val label: Block => String = _.toString
    fmt(L, label, b, false)
    b += ""
    sw.toString
  }

  def apply(x: X): String = {
    val sw = new StringWriter()
    val b = new Builder(0, sw)
    val label: Block => String = _.toString
    fmt(x, label, b, false)
    b += ""
    sw.toString
  }

  def fmt(c: Classx[_], b: Builder, saveLineNumbers: Boolean): Unit = {
    // FIXME interfaces
    if (b.printSourceLineNumbers) {
      c.sourceFile.foreach { sf =>
        b += s"source file: ${ sf }"
      }
    }
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
        fmt(x, label, b, saveLineNumbers)
        x = x.next
      }
    }
  }

  def fmt(x: X, label: Block => String, b: Builder, saveLineNumbers: Boolean): Unit = {
    val cl = x.getClass.getSimpleName
    val h = header(x, label)
    if (h != "")
      b.appendWithSource(s"($cl $h", x.lineNumber)
    else
      b.appendWithSource(s"($cl", x.lineNumber)
    if (saveLineNumbers)
      x.lineNumber = b.lineNumber
    b.indent {
      for (c <- x.children) {
        if (c != null)
          fmt(c, label, b, saveLineNumbers)
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
    case x: LdcX =>
      val lit = x.a match {
        case s: String => s""""${escapeString(s)}""""
        case a => a.toString
      }
      s"$lit ${ x.ti }"
    case x: InsnX => asm.util.Printer.OPCODES(x.op)
    case x: StoreX => x.l.toString
    case x: IincX => s"${ x.l.toString } ${ x.i }"
    case x: PutFieldX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.f }"
    case x: GetFieldX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.f }"
    case x: NewInstanceX => s"${ x.ti.iname } ${ x.ctor }"
    case x: TypeInsnX =>
      s"${ asm.util.Printer.OPCODES(x.op) } ${ x.ti.iname }"
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

package is.hail.lir

import is.hail.asm4s.{BooleanInfo, IntInfo, TypeInfo, UnitInfo}
import is.hail.utils.FastIndexedSeq
import org.objectweb.asm.Opcodes._

object SplitMethod {
  val TargetMethodSize: Int = 2000

  def apply(c: Classx[_], m: Method): Unit = {
    new SplitMethod(c, m).split()
  }
}

class SplitMethod(c: Classx[_], m: Method) {
  private val blocks = m.findBlocks()

  private val paramFields = m.parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
    c.newField(genName("f", s"arg$i"), ti)
  }

  def splitLargeStatements(): Unit = {
    for (b <- blocks) {
      def splitLargeStatement(c: StmtX): Unit = {
        def visit(x: ValueX): Int = {
          // FIXME this doesn't handle many moderate-sized children
          val size = 1 + x.children.map(visit).sum

          if (size > SplitMethod.TargetMethodSize / 2) {
            val l = m.newLocal("spill_large_expr", x.ti)
            x.replace(load(l))
            c.insertBefore(store(l, x))
            1
          } else
            size
        }

        c.children.foreach(visit)
      }

      var x = b.first
      while (x != null) {
        splitLargeStatement(x)
        x = x.next
      }
    }
  }

  def spillLocals(): Unit = {
    val (locals, localIdx) = m.findAndIndexLocals(blocks)

    val fields = locals.map { l =>
      if (l.isInstanceOf[Parameter])
        null
      else
        c.newField(genName("f", l.name), l.ti)
    }

    def localField(l: Local): Field =
      l match {
        case p: Parameter =>
          if (p.i == 0)
            null
          else
            paramFields(p.i - 1)
        case _ => fields(localIdx(l))
      }

    def spill(x: X): Unit = {
      x.children.foreach(spill)
      x match {
        case x: LoadX =>
          val f = localField(x.l)
          if (f != null)
            x.replace(getField(f, load(m.getParam(0))))
        case x: IincX =>
          val f = localField(x.l)
          assert(f != null)
          x.replace(
            putField(f, load(m.getParam(0)),
              insn(IADD,
                getField(f, load(m.getParam(0))),
                ldcInsn(x.i, IntInfo))))
        case x: StoreX =>
          val f = localField(x.l)
          assert(f != null)
          val v = x.children(0)
          v.remove()
          x.replace(putField(f, load(m.getParam(0)), v))
        case _ =>
      }
    }

    for (b <- blocks) {
      var x = b.first
      while (x != null) {
        val n = x.next
        spill(x)
        x = n
      }
    }
  }

  def splitBlock(b: Block): Unit = {
    val last = b.last

    val returnTI = last match {
      case _: GotoX => UnitInfo
      case _: IfX => BooleanInfo
      case _: SwitchX => IntInfo
      case _: ReturnX => m.returnTypeInfo
      case _: ThrowX => UnitInfo
    }

    var L = new Block()
    var x = b.first
    var size = 0

    while (x != last) {
      if (size > SplitMethod.TargetMethodSize) {
        val newM = c.newMethod(genName("m", "wrapped"), FastIndexedSeq.empty[TypeInfo[_]], UnitInfo)
        L.method = newM
        newM.setEntry(L)
        L.append(returnx())

        x.insertBefore(methodStmt(INVOKEVIRTUAL, newM, Array(load(m.getParam(0)))))

        L = new Block()
        size = 0
      }

      size += x.approxByteCodeSize()
      val n = x.next
      x.remove()
      L.append(x)
      x = n
    }

    val newM = c.newMethod(genName("m", "wrapped"), FastIndexedSeq.empty[TypeInfo[_]], returnTI)
    L.method = newM
    newM.setEntry(L)

    def invokeNewM(): ValueX =
      methodInsn(INVOKEVIRTUAL, newM, Array(load(m.getParam(0))))

    def invokeNewMStmt(): StmtX =
      methodStmt(INVOKEVIRTUAL, newM, Array(load(m.getParam(0))))

    last match {
      case _: GotoX | _: ThrowX =>
        L.append(returnx())
        last.insertBefore(invokeNewMStmt())
      case x: IfX =>
        val Ltrue = x.Ltrue
        val Lfalse = x.Lfalse

        x.remove()
        L.append(x)

        val newLtrue = new Block()
        newLtrue.method = newM
        newLtrue.append(returnx(ldcInsn(0, BooleanInfo)))
        x.setLtrue(newLtrue)

        val newLfalse = new Block()
        newLfalse.method = newM
        newLfalse.append(returnx(ldcInsn(0, BooleanInfo)))
        x.setLfalse(newLfalse)

        b.append(
          ifx(IFNE, invokeNewM(), Ltrue, Lfalse))
      case x: SwitchX => IntInfo
        val i = x.children(0)
        x.setChild(0, invokeNewM())
        L.append(returnx(i))
      case _: ReturnX =>
        if (returnTI eq UnitInfo) {
          L.append(returnx())
          last.insertBefore(invokeNewMStmt())
        } else {
          val c = x.children(0)
          x.setChild(0, invokeNewM())
          L.append(returnx(c))
        }
    }
  }

  def split(): Unit = {
    splitLargeStatements()
    spillLocals()

    for (b <- blocks) {
      splitBlock(b)
    }

    // this can't get split
    m.parameterTypeInfo.indices.foreach { i =>
      m.entry.prepend(putField(
        paramFields(i),
        load(m.getParam(0)),
        load(m.getParam(i + 1))))
    }
  }
}

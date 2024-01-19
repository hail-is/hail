package is.hail.lir

import is.hail.utils._

import scala.collection.mutable

import java.io.{ByteArrayOutputStream, PrintWriter}
import java.nio.charset.StandardCharsets

import org.objectweb.asm.{ClassReader, ClassVisitor, ClassWriter, Label}
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.util.{CheckClassAdapter, Textifier, TraceClassVisitor}

object Emit {
  def emitMethod(cv: ClassVisitor, m: Method, debugInformation: Boolean): Int = {
    val blocks = m.findBlocks()
    val static = if (m.isStatic) ACC_STATIC else 0

    val mv = cv.visitMethod(ACC_PUBLIC | static, m.name, m.desc, null, null)
    mv.visitCode()

    val labels = blocks.map(L => L -> new Label).toMap

    val localIndex: mutable.Map[Local, Int] = mutable.Map[Local, Int]()

    val start = new Label
    val end = new Label

    var nLocals = 0
    val parameterIndex = new Array[Int](m.parameterTypeInfo.length + (!m.isStatic).toInt)
    var i = 0
    while (i < parameterIndex.length) {
      parameterIndex(i) = nLocals
      if (i == 0 && !m.isStatic)
        nLocals += 1 // this
      else
        nLocals += m.parameterTypeInfo(i - (!m.isStatic).toInt).slots
      i += 1
    }

    val locals = m.findLocals(blocks)

    for (l <- locals) {
      if (!l.isInstanceOf[Parameter]) {
        localIndex += (l -> nLocals)

        // println(s"  assign $l $nLocals ${ l.ti.desc }")
        nLocals += l.ti.slots
      }
    }

    def getLocalIndex(l: Local): Int =
      l match {
        case p: Parameter => parameterIndex(p.i)
        case _ => localIndex(l)
      }

    var maxStack = 0
    var curLineNumber = -1
    var instructionCount = 0

    def setLineNumber(n: Int): Unit = {
      if (debugInformation && n != curLineNumber) {
        curLineNumber = n
        val L = new Label()
        mv.visitLabel(L)
        mv.visitLineNumber(curLineNumber, L)
      }
    }

    def emitX(x: X, depth: Int): Unit = {
      x match {
        case x: NewInstanceX =>
          setLineNumber(x.lineNumber)
          mv.visitTypeInsn(NEW, x.ti.iname)
          mv.visitInsn(DUP)

          var i = 0
          while (i < x.children.length) {
            emitX(x.children(i), depth + 2 + i)
            i += 1
          }

          if (depth + 2 > maxStack)
            maxStack = depth + 2

          setLineNumber(x.lineNumber)
          mv.visitMethodInsn(
            INVOKESPECIAL,
            x.ctor.owner,
            x.ctor.name,
            x.ctor.desc,
            x.ctor.isInterface,
          )
          instructionCount += 3
          return
        case _ =>
      }

      var i = 0
      while (i < x.children.length) {
        emitX(x.children(i), depth + i)
        i += 1
      }

      setLineNumber(x.lineNumber)

      if (depth + 1 > maxStack)
        maxStack = depth + 1

      instructionCount += 1
      x match {
        case x: IfX =>
          instructionCount += 1
          mv.visitJumpInsn(x.op, labels(x.Ltrue))
          mv.visitJumpInsn(GOTO, labels(x.Lfalse))
        case x: GotoX =>
          mv.visitJumpInsn(GOTO, labels(x.L))
        case x: SwitchX =>
          assert(x.Lcases.nonEmpty)
          mv.visitTableSwitchInsn(
            0,
            x.Lcases.length - 1,
            labels(x.Ldefault),
            x.Lcases.map(labels): _*
          )
        case x: ReturnX =>
          if (x.children.length == 0)
            mv.visitInsn(RETURN)
          else
            mv.visitInsn(m.returnTypeInfo.returnOp)
        case x: LoadX =>
          mv.visitVarInsn(x.l.ti.loadOp, getLocalIndex(x.l))
        case x: StoreX =>
          mv.visitVarInsn(x.l.ti.storeOp, getLocalIndex(x.l))
        case x: PutFieldX =>
          mv.visitFieldInsn(x.op, x.f.owner, x.f.name, x.f.ti.desc)
        case x: InsnX =>
          mv.visitInsn(x.op)
        case x: TypeInsnX =>
          mv.visitTypeInsn(x.op, x.ti.iname)
        case x: MethodX =>
          mv.visitMethodInsn(
            x.op,
            x.method.owner,
            x.method.name,
            x.method.desc,
            x.method.isInterface,
          )
        case x: MethodStmtX =>
          mv.visitMethodInsn(
            x.op,
            x.method.owner,
            x.method.name,
            x.method.desc,
            x.method.isInterface,
          )
        case x: LdcX =>
          mv.visitLdcInsn(x.a)
        case x: GetFieldX =>
          mv.visitFieldInsn(x.op, x.f.owner, x.f.name, x.f.ti.desc)
        case x: NewArrayX =>
          x.eti.newArray().accept(mv)
        case x: IincX =>
          mv.visitIincInsn(getLocalIndex(x.l), x.i)
        case x: StmtOpX =>
          mv.visitInsn(x.op)
        case _: ThrowX =>
          mv.visitInsn(ATHROW)
      }
    }

    def emitBlock(L: Block): Unit = {
      mv.visitLabel(labels(L))
      var x = L.first
      while (x != null) {
        emitX(x, 0)
        x = x.next
      }
    }

    mv.visitLabel(start)

    emitBlock(m.entry)
    for (b <- blocks)
      if (b ne m.entry)
        emitBlock(b)

    mv.visitLabel(end)

    for (l <- locals)
      if (!l.isInstanceOf[Parameter]) {
        val n = localIndex(l)
        val name = if (l.name == null) s"local$n" else l.name
        mv.visitLocalVariable(name, l.ti.desc, null, start, end, n)
      }

    mv.visitMaxs(maxStack, nLocals)

    mv.visitEnd()

    instructionCount
  }

  def emitClass(c: Classx[_], cv: ClassVisitor, logMethodSizes: Boolean): Unit = {
    cv.visit(V1_8, ACC_PUBLIC, c.name, null, c.superName, c.interfaces.toArray)
    c.sourceFile.foreach(cv.visitSource(_, null))

    for (f <- c.fields)
      f match {
        case f: Field => cv.visitField(ACC_PUBLIC, f.name, f.ti.desc, null, null)
        case f: StaticField => cv.visitField(ACC_PUBLIC | ACC_STATIC, f.name, f.ti.desc, null, null)
      }

    for (m <- c.methods) {
      val instructionCount = emitMethod(cv, m, c.sourceFile.isDefined)
      if (logMethodSizes) {
        log.info(s"instruction count: $instructionCount: ${c.name}.${m.name}")
        if (instructionCount > 8000)
          log.warn(s"big method: $instructionCount: ${c.name}.${m.name}")
      }
    }

    cv.visitEnd()
  }

  def apply(c: Classx[_], print: Option[PrintWriter]): Array[Byte] = {
    val bytes =
      try {
        val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS + ClassWriter.COMPUTE_FRAMES)

        emitClass(c, cw, logMethodSizes = true)

        val b = cw.toByteArray
        // For efficiency, the ClassWriter does no checking, and may generate invalid
        // bytecode. This will verify the generated class file, printing errors
        // to System.out.
        // This next line should always be commented out!
//      CheckClassAdapter.verify(new ClassReader(b), false, new PrintWriter(System.err))
        b
      } catch {
        case e: Exception =>
          val buffer = new ByteArrayOutputStream()
          val trace = new TraceClassVisitor(new PrintWriter(buffer))
          val check = new CheckClassAdapter(trace)
          val classJVMByteCodeAsEscapedStr = buffer.toString(StandardCharsets.UTF_8.name())
          log.error(s"lir exception $e:\n" + classJVMByteCodeAsEscapedStr)
          emitClass(c, check, logMethodSizes = false)
          throw e
      }
    print.foreach { pw =>
      val cr = new ClassReader(bytes)
      val tcv = new TraceClassVisitor(null, new Textifier, pw)
      cr.accept(tcv, 0)
    }
    bytes
  }
}

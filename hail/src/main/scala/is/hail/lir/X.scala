package is.hail.lir

import is.hail.asm4s._
import is.hail.utils._
import org.objectweb.asm.Opcodes._

import java.io.PrintWriter
import scala.collection.mutable

// FIXME move typeinfo stuff lir

class Classx[C](val name: String, val superName: String, var sourceFile: Option[String]) {
  val ti: TypeInfo[C] = new ClassInfo[C](name)

  val methods: mutable.ArrayBuffer[Method] = new mutable.ArrayBuffer()

  val fields: mutable.ArrayBuffer[FieldRef] = new mutable.ArrayBuffer()

  val interfaces: mutable.ArrayBuffer[String] = new mutable.ArrayBuffer()

  def addInterface(name: String): Unit = {
    interfaces += name
  }

  def newField(name: String, ti: TypeInfo[_]): Field = {
    val f = new Field(this, name, ti)
    fields += f
    f
  }

  def genField(baseName: String, ti: TypeInfo[_]): Field = newField(genName("f", baseName), ti)

  def newStaticField(name: String, ti: TypeInfo[_]): StaticField = {
    val f = new StaticField(this, name, ti)
    fields += f
    f
  }

  def newMethod(name: String,
    parameterTypeInfo: IndexedSeq[TypeInfo[_]],
    returnTypeInfo: TypeInfo[_],
    isStatic: Boolean = false): Method = {
    val method = new Method(this, name, parameterTypeInfo, returnTypeInfo, isStatic)
    methods += method
    method
  }

  def saveToFile(path: String): Unit = {
    val file = new java.io.File(path)
    file.getParentFile.mkdirs()
    using (new java.io.PrintWriter(file)) { out =>
      Pretty(this, out, saveLineNumbers = true)
    }
    sourceFile = Some(path)
  }

  def asBytes(writeIRs: Boolean, print: Option[PrintWriter]): Array[(String, Array[Byte])] = {
    val classes = new mutable.ArrayBuffer[Classx[_]]()
    classes += this

    for (m <- methods) {
      m.verify()
      SimplifyControl(m)
    }

    val shortName = name.take(50)
    if (writeIRs) saveToFile(s"/tmp/hail/${shortName}.lir")

    for (m <- methods) {
      if (m.name != "<init>"
      && m.approxByteCodeSize() > SplitMethod.TargetMethodSize
      ) {
        SplitLargeBlocks(m)

        val blocks = m.findBlocks()
        val locals = m.findLocals(blocks)

        val PSTResult(blocks2, cfg2, pst) = {
          // this cfg is no longer valid after creating pst
          val cfg = CFG(m, blocks)
          PST(m, blocks, cfg)
        }

        val liveness = Liveness(blocks2, locals, cfg2)

        classes += SplitMethod(this, m, blocks2, locals, cfg2, liveness, pst)

        // clean up after SplitMethod
        SimplifyControl(m)
      }
    }

    for (m <- methods) {
      val blocks = m.findBlocks()
      val locals = m.findLocals(blocks)
      val cfg = CFG(m, blocks)
      val liveness = Liveness(blocks, locals, cfg)

      InitializeLocals(m, blocks, locals, liveness)
    }

    if (writeIRs) saveToFile(s"/tmp/hail/${shortName}.split.lir")

    // println(Pretty(this, saveLineNumbers = false))
    classes.iterator.map { c =>
      val bytes = Emit(c,
        print
        // Some(new PrintWriter(System.out))
      )

      if (writeIRs) {
        val classFile = new java.io.File(s"/tmp/hail/${c.name.take(50)}.class")
        classFile.getParentFile.mkdirs()
        using (new java.io.FileOutputStream(classFile)) { fos =>
          fos.write(bytes)
        }
      }

      (c.name.replace("/", "."), bytes)
    }.toArray
  }
}

abstract class FieldRef {
  def owner: String

  def name: String

  def ti: TypeInfo[_]

  override def toString: String = s"$owner.$name ${ ti.desc }"
}

class Field private[lir] (classx: Classx[_], val name: String, val ti: TypeInfo[_]) extends FieldRef {
  def owner: String = classx.name
}

class StaticField private[lir] (classx: Classx[_], val name: String, val ti: TypeInfo[_]) extends FieldRef {
  def owner: String = classx.name
}

class FieldLit(
  val owner: String,
  val name: String,
  val ti: TypeInfo[_]) extends FieldRef

abstract class MethodRef {
  def owner: String

  def name: String

  def desc: String

  def isInterface: Boolean

  def returnTypeInfo: TypeInfo[_]

  override def toString: String =
    s"$owner.$name $desc${ if (isInterface) "interface" else "" }"
}

class Method private[lir] (
  val classx: Classx[_],
  val name: String,
  val parameterTypeInfo: IndexedSeq[TypeInfo[_]],
  val returnTypeInfo: TypeInfo[_],
  val isStatic: Boolean) extends MethodRef {

  def nParameters: Int = parameterTypeInfo.length + (!isStatic).toInt

  def owner: String = classx.name

  def desc = s"(${parameterTypeInfo.map(_.desc).mkString})${returnTypeInfo.desc}"

  def isInterface: Boolean = false

  private var _entry: Block = _

  def setEntry(newEntry: Block): Unit = {
    _entry = newEntry
  }

  def entry: Block = _entry

  def getParam(i: Int): Parameter = {
    new Parameter(this, i,
      if (i == 0 && !isStatic)
        new ClassInfo(classx.name)
      else
        parameterTypeInfo(i - (!isStatic).toInt))
  }

  def newLocal(name: String, ti: TypeInfo[_]): Local =
    new Local(this, name, ti)

  def genLocal(baseName: String, ti: TypeInfo[_]): Local = newLocal(genName("l", baseName), ti)

  def findBlocks(): Blocks = {
    val blocksb = new BoxedArrayBuilder[Block]()

    val s = new ObjectArrayStack[Block]()
    val visited = mutable.Set[Block]()

    s.push(entry)

    while (s.nonEmpty) {
      val L = s.pop()
      if (!visited.contains(L)) {
        assert(L.wellFormed)

        if (L.method == null)
          L.method = this
        else {
          /*
          if (L.method ne this) {
            println(s"${ L.method } $this")
            // println(b.stack.mkString("\n"))
          }
           */
          assert(L.method eq this)
        }

        blocksb += L

        assert(L.first != null)
        val x = L.last.asInstanceOf[ControlX]
        var i = x.targetArity() - 1
        while (i >= 0) {
          val target = x.target(i)
          assert(target != null)
          s.push(target)
          i -= 1
        }
        visited += L
      }
    }

    val blocks = blocksb.result()

    // prune dead Block uses
    for (b <- blocks) {
      // don't traverse a set that's being modified
      val uses2 = b.uses.toArray
      for ((u, i) <- uses2) {
        if (u.parent == null || !visited(u.parent))
          u.setTarget(i, null)
      }
    }

    new Blocks(blocks)
  }

  def findLocals(blocks: Blocks, verifyMethodAssignment: Boolean = false): Locals = {
    val localsb = new BoxedArrayBuilder[Local]()

    var i = 0
    while (i < nParameters) {
      localsb += (
        if (i == 0 && !isStatic)
          new Parameter(this, 0, classx.ti)
        else
          new Parameter(this, i, parameterTypeInfo(i - (!isStatic).toInt)))
      i += 1
    }

    val visited: mutable.Set[Local] = mutable.Set()

    def visitLocal(l: Local): Unit = {
      if (!l.isInstanceOf[Parameter]) {
        if (!visited.contains(l)) {
          if (!verifyMethodAssignment || l.method == null)
            l.method = this
          else {
            if (l.method ne this) {
              println(s"$l ${l.method} ${this}\n  ${l.stack.mkString("  \n")}")
              println(s"$l ${l.method} ${this}")
            }

            assert(l.method eq this)
          }

          localsb += l
          visited += l
        }
      }
    }

    def visitX(x: X): Unit = {
      x match {
        case x: StoreX => visitLocal(x.l)
        case x: LoadX => visitLocal(x.l)
        case x: IincX => visitLocal(x.l)
        case _ =>
      }
      x.children.foreach(visitX)
    }

    for (b <- blocks) {
      var x = b.first
      while (x != null) {
        visitX(x)
        x = x.next
      }
    }

    new Locals(localsb.result())
  }

  // Verify all blocks are well-formed, all blocks and locals have correct
  // method set.
  def verify(): Unit = {
    findLocals(findBlocks(), verifyMethodAssignment = true)
  }

  def approxByteCodeSize(): Int = {
    val blocks = findBlocks()
    var size = 0
    for (b <- blocks) {
      size += b.approxByteCodeSize()
    }
    size
  }
}

class MethodLit(
  val owner: String, val name: String, val desc: String, val isInterface: Boolean,
  val returnTypeInfo: TypeInfo[_]
) extends MethodRef

class Local(var method: Method, val name: String, val ti: TypeInfo[_]) {
  override def toString: String = f"t${ System.identityHashCode(this) }%08x/$name ${ ti.desc }"
  val stack = Thread.currentThread().getStackTrace
}

class Parameter(method: Method, val i: Int, ti: TypeInfo[_]) extends Local(method, null, ti) {
  override def toString: String = s"arg:$i ${ ti.desc }"
}

class Block {
  // for debugging
  // val stack = Thread.currentThread().getStackTrace.mkString("\n")

  var method: Method = _

  var first: StmtX = _
  var last: StmtX = _

  val uses: mutable.Set[(ControlX, Int)] = mutable.Set[(ControlX, Int)]()

  def wellFormed: Boolean = {
    if (first == null)
      return false

    last match {
      case ctrl: ControlX =>
        (0 until ctrl.targetArity()).forall(ctrl.target(_) != null)
      case _ => false
    }
  }

  def addUse(x: ControlX, i: Int): Unit = {
    val added = uses.add(x -> i)
    assert(added)
  }

  def removeUse(x: ControlX, i: Int): Unit = {
    val removed = uses.remove(x -> i)
    assert(removed)
  }

  def replace(L: Block): Unit = {
    if (method.entry eq this)
      method.setEntry(L)

    // don't traverse a set that's being modified
    val uses2 = uses.toArray
    for ((x, i) <- uses2) {
      x.setTarget(i, L)
    }
    assert(uses.isEmpty)
  }

  def prepend(x: StmtX): Unit = {
    assert(x.parent == null)
    if (x.isInstanceOf[ControlX])
      // prepending a new control statement, so previous contents are dead code
      while (last != null) {
        last.remove()
      }
    if (last == null) {
      first = x
      last = x
    } else {
      assert(x.next == null)
      x.next = first
      assert(first.prev == null)
      first.prev = x
      first = x
    }
    x.parent = this
  }

  def append(x: StmtX): Unit = {
    assert(x.parent == null)
    assert(!last.isInstanceOf[ControlX], s"StmtX '$x' is redundant after ControlX '$last'.")

    if (last == null) {
      first = x
      last = x
    } else {
      assert(x.prev == null)
      x.prev = last
      assert(last.next == null)
      last.next = x
      last = x
    }
    x.parent = this
  }

  def drop(): Unit = {
    var t = first
    while (t != null) {
      val n = t.next
      t.next = null
      t.prev = null
      t.parent = null
      t = n
    }
    first = null
    last = null
  }

  override def toString: String = f"L${ System.identityHashCode(this) }%08x"

  def approxByteCodeSize(): Int = {
    var size = 1 // for the block
    var x = first
    while (x != null) {
      size += x.approxByteCodeSize()
      x = x.next
    }
    size
  }

}

// X stands for eXpression
abstract class X {
  // for debugging
  // val stack = Thread.currentThread().getStackTrace
  // var setParentStack: Array[StackTraceElement] = _

  var children: Array[ValueX] = new Array(0)

  var lineNumber: Int

  def setArity(n: Int): Unit = {
    var i = n
    while (i < children.length) {
      val c = children(i)
      c.parent = null
      children(i) = null
      i += 1
    }
    children = java.util.Arrays.copyOf(children, n)
  }

  def setChild(i: Int, x: ValueX): Unit = {
    val c = children(i)
    if (c != null)
      c.parent = null

    if (x != null) {
      /*
      if (x.parent != null) {
        println(x.setParentStack.mkString("\n"))
        println("-------")
        println(x.stack.mkString("\n"))
      }
       */
      assert(x.parent == null)
      x.parent = this
      // x.setParentStack = Thread.currentThread().getStackTrace
    }
    children(i) = x
  }

  def remove(): Unit

  def containingBlock(): Block = {
    var x: X = this
    while (x != null) {
      x match {
        case vx: ValueX =>
          x = vx.parent
        case sx: StmtX =>
          return sx.parent
      }
    }
    null
  }

  def containingMethod(): Method = {
    val L = containingBlock()
    if (L != null)
      L.method
    else
      null
  }

  def approxByteCodeSize(): Int = {
    var size = 0
    def visit(x: X): Unit = {
      size += 1
      x.children.foreach(visit)
    }
    visit(this)
    size
  }
}

abstract class StmtX extends X {
  var parent: Block = _

  var prev: StmtX = _
  var next: StmtX = _

  def remove(): Unit = {
    assert(parent != null)
    if (parent.first == this)
      parent.first = next
    if (parent.last == this)
      parent.last = prev
    if (next != null)
      next.prev = prev
    if (prev != null)
      prev.next = next

    parent = null
    next = null
    prev = null
  }

  def replace(x: StmtX): Unit = {
    assert(x.parent == null)
    assert(parent != null)

    x.next = next
    x.prev = prev
    x.parent = parent

    if (parent.first == this)
      parent.first = x
    if (parent.last == this)
      parent.last = x
    if (next != null)
      next.prev = x
    if (prev != null)
      prev.next = x

    next = null
    prev = null
    parent = null
  }

  def insertBefore(x: StmtX): Unit = {
    assert(parent != null)
    assert(x.parent == null)

    x.next = this
    x.parent = parent
    if (prev != null) {
      x.prev = prev
      prev.next = x
    } else {
      parent.first = x
    }
    prev = x
  }

  def insertAfter(x: StmtX): Unit = {
    assert(parent != null)
    assert(x.parent == null)

    x.prev = this
    x.parent = parent
    if (next != null) {
      x.next = next
      next.prev = x
    } else {
      parent.last = x
    }
    next = x
  }
}

abstract class ControlX extends StmtX {
  def targetArity(): Int

  def target(i: Int): Block

  def setTarget(i: Int, b: Block): Unit
}

abstract class ValueX extends X {
  var parent: X = _

  def ti: TypeInfo[_]

  def remove(): Unit = {
    var i = 0
    while (parent.children(i) ne this)
      i += 1
    parent.setChild(i, null)
    assert(parent == null)
  }

  def replace(x: ValueX): Unit = {
    var i = 0
    while (parent.children(i) ne this)
      i += 1
    parent.setChild(i, x)
    assert(parent == null)
  }
}

class GotoX(var lineNumber: Int = 0) extends ControlX {
  private var _L: Block = _

  def L: Block = _L

  def setL(newL: Block): Unit = setTarget(0, newL)

  def targetArity(): Int = 1

  def target(i: Int): Block = {
    assert(i == 0)
    _L
  }

  def setTarget(i: Int, b: Block): Unit = {
    assert(i == 0)
    if (_L != null)
      _L.removeUse(this, 0)
    _L = b
    if (b != null)
      b.addUse(this, 0)
  }
}

class IfX(val op: Int, var lineNumber: Int = 0) extends ControlX {
  private var _Ltrue: Block = _
  private var _Lfalse: Block = _

  def Ltrue: Block = _Ltrue

  def Lfalse: Block = _Lfalse

  def setLtrue(newLtrue: Block): Unit = setTarget(0, newLtrue)

  def setLfalse(newLfalse: Block): Unit = setTarget(1, newLfalse)

  def targetArity(): Int = 2

  def target(i: Int): Block = {
    if (i == 0)
      _Ltrue
    else {
      assert(i == 1)
      _Lfalse
    }
  }

  def setTarget(i: Int, b: Block): Unit = {
    if (i == 0) {
      if (_Ltrue != null)
        _Ltrue.removeUse(this, 0)
      _Ltrue = b
      if (b != null)
        b.addUse(this, 0)
    } else {
      assert(i == 1)
      if (_Lfalse != null)
        _Lfalse.removeUse(this, 1)
      _Lfalse = b
      if (b != null)
        b.addUse(this, 1)
    }
  }
}

class SwitchX(var lineNumber: Int = 0) extends ControlX {
  private var _Ldefault: Block = _

  private var _Lcases: Array[Block] = Array.empty[Block]

  def Ldefault: Block = _Ldefault

  def setLdefault(newLdefault: Block): Unit = setTarget(0, newLdefault)

  def Lcases: IndexedSeq[Block] = _Lcases

  def setLcases(newLcases: IndexedSeq[Block]): Unit = {
    for ((block, i) <- _Lcases.zipWithIndex) {
      if (block != null) block.removeUse(this, i + 1)
    }

    // don't allow sharing
    _Lcases = Array(newLcases: _*)

    for ((block, i) <- _Lcases.zipWithIndex) {
      if (block != null) block.addUse(this, i + 1)
    }
  }

  def targetArity(): Int = 1 + _Lcases.length

  def target(i: Int): Block = {
    if (i == 0)
      _Ldefault
    else
      _Lcases(i - 1)
  }

  def setTarget(i: Int, b: Block): Unit = {
    if (i == 0) {
      if (_Ldefault != null)
        _Ldefault.removeUse(this, 0)
      _Ldefault = b
      if (b != null)
        b.addUse(this, 0)
    } else {
      val L = _Lcases(i - 1)
      if (L != null)
        L.removeUse(this, i)
      _Lcases(i - 1) = b
      if (b != null)
        b.addUse(this, i)
    }
  }
}

class StoreX(var l: Local, var lineNumber: Int = 0) extends StmtX

class PutFieldX(val op: Int, val f: FieldRef, var lineNumber: Int = 0) extends StmtX

class IincX(var l: Local, val i: Int, var lineNumber: Int = 0) extends StmtX

class ReturnX(var lineNumber: Int = 0) extends ControlX {
  def targetArity(): Int = 0

  def target(i: Int): Block = throw new IndexOutOfBoundsException()

  def setTarget(i: Int, b: Block): Unit = throw new IndexOutOfBoundsException()
}

class ThrowX(var lineNumber: Int = 0) extends ControlX {
  def targetArity(): Int = 0

  def target(i: Int): Block = throw new IndexOutOfBoundsException()

  def setTarget(i: Int, b: Block): Unit = throw new IndexOutOfBoundsException()
}

class StmtOpX(val op: Int, var lineNumber: Int = 0) extends StmtX

class MethodStmtX(val op: Int, val method: MethodRef, var lineNumber: Int = 0) extends StmtX

class TypeInsnX(val op: Int, val ti: TypeInfo[_], var lineNumber: Int = 0) extends ValueX {
}

class InsnX(val op: Int, _ti: TypeInfo[_], var lineNumber: Int = 0) extends ValueX {
  def ti: TypeInfo[_] = {
    if (_ti != null)
      return _ti

    op match {
      // Int + Boolean
      case IAND =>
        children.head.ti
      case IOR =>
        children.head.ti
      case IXOR =>
        children.head.ti

      // Int
      case INEG => IntInfo
      case IADD => IntInfo
      case ISUB => IntInfo
      case IMUL => IntInfo
      case IDIV => IntInfo
      case IREM => IntInfo
      case ISHL => IntInfo
      case ISHR => IntInfo
      case IUSHR => IntInfo
      case LCMP => IntInfo
      case FCMPL => IntInfo
      case FCMPG => IntInfo
      case DCMPL => IntInfo
      case DCMPG => IntInfo
      case L2I => IntInfo
      case F2I => IntInfo
      case D2I => IntInfo
      case ARRAYLENGTH => IntInfo
      // Long
      case LNEG => LongInfo
      case LADD => LongInfo
      case LSUB => LongInfo
      case LMUL => LongInfo
      case LDIV => LongInfo
      case LREM => LongInfo
      case LAND => LongInfo
      case LOR => LongInfo
      case LXOR => LongInfo
      case LSHL => LongInfo
      case LSHR => LongInfo
      case LUSHR => LongInfo
      case I2L => LongInfo
      case F2L => LongInfo
      case D2L => LongInfo
      // Float
      case FNEG => FloatInfo
      case FADD => FloatInfo
      case FSUB => FloatInfo
      case FMUL => FloatInfo
      case FDIV => FloatInfo
      case FREM => FloatInfo
      case I2F => FloatInfo
      case L2F => FloatInfo
      case D2F => FloatInfo

      // Double
      case DNEG => DoubleInfo
      case DADD => DoubleInfo
      case DSUB => DoubleInfo
      case DMUL => DoubleInfo
      case DDIV => DoubleInfo
      case DREM => DoubleInfo
      case I2D => DoubleInfo
      case L2D => DoubleInfo
      case F2D => DoubleInfo

      // Byte
      case I2B => ByteInfo
    }
  }
}

class LoadX(var l: Local, var lineNumber: Int = 0) extends ValueX {
  def ti: TypeInfo[_] = l.ti
}

class GetFieldX(val op: Int, val f: FieldRef, var lineNumber: Int = 0) extends ValueX {
  def ti: TypeInfo[_] = f.ti
}

class NewArrayX(val eti: TypeInfo[_], var lineNumber: Int = 0) extends ValueX {
  def ti: TypeInfo[_] = arrayInfo(eti)
}

class NewInstanceX(val ti: TypeInfo[_], val ctor: MethodRef, var lineNumber: Int = 0) extends ValueX

class LdcX(val a: Any, val ti: TypeInfo[_], var lineNumber: Int = 0) extends ValueX {
  assert(
    a.isInstanceOf[String] || a.isInstanceOf[Double] || a.isInstanceOf[Float] || a.isInstanceOf[Int] || a.isInstanceOf[Long],
    s"not a string, double, float, int, or long: $a")
}

class MethodX(val op: Int, val method: MethodRef, var lineNumber: Int = 0) extends ValueX {
  def ti: TypeInfo[_] = method.returnTypeInfo
}

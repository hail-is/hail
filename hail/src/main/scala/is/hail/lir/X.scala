package is.hail.lir

import java.io.PrintWriter

import scala.collection.mutable

import is.hail.asm4s._
import is.hail.utils._

import org.objectweb.asm.Opcodes._

// FIXME move typeinfo stuff lir

class Classx[C](val name: String, val superName: String) {
  val methods: mutable.ArrayBuffer[Method] = new mutable.ArrayBuffer()

  val fields: mutable.ArrayBuffer[Field] = new mutable.ArrayBuffer()

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

  def newMethod(name: String,
    parameterTypeInfo: IndexedSeq[TypeInfo[_]],
    returnTypeInfo: TypeInfo[_]): Method = {
    val method = new Method(this, name, parameterTypeInfo, returnTypeInfo)
    methods += method
    method
  }

  def asBytes(print: Option[PrintWriter]): Array[Byte] = {
    for (m <- methods) {
      val blocks = m.findBlocks()
      for (b <- blocks) {
        if (b.method == null)
          b.method = m
        else {
          /*
        if (m.method ne this) {
          println(s"${ b.method } $m")
          println(b.stack.mkString("\n"))
        }
         */
          assert(b.method eq m)
        }
      }

      val locals = m.findLocals(blocks)
      for (l <- locals) {
        // FIXME parameters, too
        if (!l.isInstanceOf[Parameter]) {
          if (l.method == null)
            l.method = m
          else {
            /*
            if (l.method ne m) {
              println(s"$l ${l.method} $m")
            }
             */
            assert(l.method eq m)
          }
        }
      }
    }

    for (m <- methods) {
      m.removeDeadCode()
    }

    // check
    for (m <- methods) {
      val blocks = m.findBlocks()
      for (b <- blocks) {
        assert(b.first != null)
        assert(b.last.isInstanceOf[ControlX])
      }
    }

    for (m <- methods) {
      m.simplifyBlocks()
    }

    for (m <- methods) {
      InitializeLocals(m)
    }

    Emit(this,
      print
      // Some(new PrintWriter(System.out))
    )

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
  val returnTypeInfo: TypeInfo[_]) extends MethodRef {
  def owner: String = classx.name

  def desc = s"(${parameterTypeInfo.map(_.desc).mkString})${returnTypeInfo.desc}"

  def isInterface: Boolean = false

  var _entry: Block = _

  def setEntry(newEntry: Block): Unit = {
    _entry = newEntry
  }

  def entry: Block = _entry

  def getParam(i: Int): Parameter = {
    new Parameter(this, i,
      if (i == 0)
        new ClassInfo(classx.name)
      else
        parameterTypeInfo(i - 1))
  }

  def newLocal(name: String, ti: TypeInfo[_]): Local =
    new Local(this, name, ti)

  def genLocal(baseName: String, ti: TypeInfo[_]): Local = newLocal(genName("l", baseName), ti)

  def findBlocks(): IndexedSeq[Block] = {
    val blocks = mutable.ArrayBuffer[Block]()

    val s = new mutable.Stack[Block]()
    val visited = mutable.Set[Block]()

    s.push(entry)

    while (s.nonEmpty) {
      val L = s.pop()
      if (!visited.contains(L)) {
        blocks += L

        var x = L.first
        while (x != null) {
          x match {
            case x: IfX =>
              s.push(x.Ltrue)
              s.push(x.Lfalse)
            case x: GotoX =>
              s.push(x.L)
            case x: SwitchX =>
              s.push(x.Ldefault)
              x.Lcases.foreach(s.push)
            case _ =>
          }
          x = x.next
        }
        visited += L
      }
    }

    blocks
  }

  def findAndIndexBlocks(): (IndexedSeq[Block], Map[Block, Int]) = {
    val blocks = findBlocks()
    val blockIdx = blocks.zipWithIndex.toMap
    (blocks, blockIdx)
  }

  def findLocals(blocks: IndexedSeq[Block]): IndexedSeq[Local] = {
    val locals = mutable.ArrayBuffer[Local]()
    val visited: mutable.Set[Local] = mutable.Set()

    def visitLocal(l: Local): Unit = {
      if (!visited.contains(l)) {
        locals += l
        visited += l
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

    locals
  }

  def findAndIndexLocals(blocks: IndexedSeq[Block]): (IndexedSeq[Local], Map[Local, Int]) = {
    val locals = findLocals(blocks)
    val localIdx = locals.zipWithIndex.toMap
    (locals, localIdx)
  }

  def removeDeadCode(): Unit = {
    val blocks = findBlocks()
    for (b <- blocks) {
      var x = b.first
      while (x != null && !x.isInstanceOf[ControlX])
        x = x.next
      if (x != null) {
        while (x.next != null)
          x.next.remove()
      }
    }
  }

  def simplifyBlocks(): Unit = {
    @scala.annotation.tailrec
    def finalTarget(b: Block): Block = {
      if (b.first != null &&
        b.first.isInstanceOf[GotoX]) {
        finalTarget(b.first.asInstanceOf[GotoX].L)
      } else
        b
    }

    @scala.annotation.tailrec
    def simplifyBlock(L: Block): Unit = {
      L.last match {
        case x: IfX =>
          x.setLtrue(finalTarget(x.Ltrue))
          x.setLfalse(finalTarget(x.Lfalse))

          val M = x.Ltrue
          if (x.Lfalse eq M) {
            x.remove()
            x.setLtrue(null)
            x.setLfalse(null)
            L.append(goto(M))
          }

        case x: GotoX =>
          val M = x.L
          if ((M ne L) && M.uses.size == 1 && (entry ne M)) {
            x.remove()
            x.setL(null)
            while (M.first != null) {
              val z = M.first
              z.remove()
              L.append(z)
            }
            simplifyBlock(L)
          } else
            x.setL(finalTarget(x.L))

        case _ =>
      }
    }

    val blocks = findBlocks()

    for (ell <- blocks)
      simplifyBlock(ell)
  }
}

class MethodLit(
  val owner: String, val name: String, val desc: String, val isInterface: Boolean,
  val returnTypeInfo: TypeInfo[_]
) extends MethodRef

class Local(var method: Method, val name: String, val ti: TypeInfo[_]) {
  override def toString: String = f"t${ System.identityHashCode(this) }%08x/$name"
}

class Parameter(method: Method, val i: Int, ti: TypeInfo[_]) extends Local(method, null, ti) {
  override def toString: String = s"arg:$i"
}

class Block {
  // for debugging
  // val stack = Thread.currentThread().getStackTrace

  var method: Method = _

  val uses: mutable.Set[ControlX] = mutable.Set()

  var first: StmtX = _
  var last: StmtX = _

  def replace(L: Block): Unit = {
    if (method != null && (method.entry eq this)) {
      method.setEntry(L)
    }

    while (uses.nonEmpty) {
      val x = uses.head
      x match {
        case x: GotoX =>
          assert(x.L eq this)
          x.setL(L)
        case x: IfX =>
          if (x.Ltrue eq this)
            x.setLtrue(L)
          if (x.Lfalse eq this)
            x.setLfalse(L)
      }
    }

    assert(uses.isEmpty)
  }

  def prepend(x: StmtX): Unit = {
    assert(x.parent == null)
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
}

// X stands for eXpression
abstract class X {
  // for debugging
  // val stack = Thread.currentThread().getStackTrace
  // var setParentStack: Array[StackTraceElement] = _

  var children: Array[ValueX] = new Array(0)

  def setArity(n: Int): Unit = {
    var i = n
    while (i < children.length) {
      val c = children(i)
      c.parent = null
      children(i) = null
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
}

abstract class ControlX extends StmtX

abstract class ValueX extends X {
  var parent: X = _

  def ti: TypeInfo[_]
}

class GotoX extends ControlX {
  var _L: Block = _

  def L: Block = _L

  def setL(newL: Block): Unit = {
    if (_L != null)
      _L.uses -= this
    if (newL != null)
      newL.uses += this
    _L = newL
  }
}

class IfX(val op: Int) extends ControlX {
  var _Ltrue: Block = _

  def Ltrue: Block = _Ltrue

  def setLtrue(newLtrue: Block): Unit = {
    removeUses()
    _Ltrue = newLtrue
    addUses()
  }

  var _Lfalse: Block = _

  def Lfalse: Block = _Lfalse

  def setLfalse(newLfalse: Block): Unit = {
    removeUses()
    _Lfalse = newLfalse
    addUses()
  }

  private def removeUses(): Unit = {
    if (_Ltrue != null)
      _Ltrue.uses -= this
    if (_Lfalse != null)
      _Lfalse.uses -= this
  }

  private def addUses(): Unit = {
    if (_Ltrue != null)
      _Ltrue.uses += this
    if (_Lfalse != null)
      _Lfalse.uses += this
  }
}

class SwitchX() extends ControlX {
  private[this] var _Ldefault: Block = _

  private[this] var _Lcases: IndexedSeq[Block] = FastIndexedSeq()

  def Ldefault: Block = _Ldefault

  def Lcases: IndexedSeq[Block] = _Lcases

  def setDefault(newDefault: Block): Unit = {
    if (_Ldefault != null)
      _Ldefault.uses -= this
    _Ldefault = newDefault
    if (_Ldefault != null)
      _Ldefault.uses += this
  }

  def setCases(newCases: IndexedSeq[Block]): Unit = {
    _Lcases.foreach { c =>
      if (c != null)
        c.uses -= this
    }
    _Lcases = newCases
    _Lcases.foreach { c =>
      if (c != null)
        c.uses += this
    }
  }
}

class StoreX(val l: Local) extends StmtX

class PutFieldX(val op: Int, val f: FieldRef) extends StmtX

class IincX(val l: Local, val i: Int) extends StmtX

class ReturnX() extends ControlX

class ThrowX() extends ControlX

class StmtOpX(val op: Int) extends StmtX

class MethodStmtX(val op: Int, val method: MethodRef) extends StmtX

class TypeInsnX(val op: Int, val t: String) extends ValueX {
  def ti: TypeInfo[_] = {
    assert(op == CHECKCAST)
    // FIXME, ClassInfo should take the internal name
    new ClassInfo(t.replace("/", "."))
  }
}

class InsnX(val op: Int, _ti: TypeInfo[_]) extends ValueX {
  def ti: TypeInfo[_] = {
    if (_ti != null)
      return _ti

    op match {
      // Int
      case INEG => IntInfo
      case IADD => IntInfo
      case ISUB => IntInfo
      case IMUL => IntInfo
      case IDIV => IntInfo
      case IREM => IntInfo
      case IAND => IntInfo
      case IOR => IntInfo
      case IXOR => IntInfo
      case L2I => IntInfo
      case F2I => IntInfo
      case D2I => IntInfo
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
    }
  }
}

class LoadX(val l: Local) extends ValueX {
  def ti: TypeInfo[_] = l.ti
}

class GetFieldX(val op: Int, val f: FieldRef) extends ValueX {
  def ti: TypeInfo[_] = f.ti
}

class NewArrayX(val eti: TypeInfo[_]) extends ValueX {
  def ti: TypeInfo[_] = arrayInfo(eti)
}

class NewInstanceX(val ti: TypeInfo[_]) extends ValueX

class LdcX(val a: Any) extends ValueX {
  val ti: TypeInfo[_] = a match {
    case _: Boolean => BooleanInfo
    case _: Byte => ByteInfo
    case _: Short => ShortInfo
    case _: Int => IntInfo
    case _: Long => LongInfo
    case _: Char => CharInfo
    case _: Float => FloatInfo
    case _: Double => DoubleInfo
    case _: String => classInfo[String]
  }
}

class MethodX(val op: Int, val method: MethodRef) extends ValueX {
  def ti: TypeInfo[_] = method.returnTypeInfo
}

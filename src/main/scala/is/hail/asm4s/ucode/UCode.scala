package is.hail.asm4s.ucode

import is.hail.asm4s._

import is.hail.annotations._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.language.existentials

trait UCode {
  def emit(il: Growable[AbstractInsnNode]): Unit
}

abstract class InsnUCode(insn: AbstractInsnNode) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) { il += insn }
}

case class Null() extends InsnUCode(new InsnNode(ACONST_NULL))
case class I32(x: Int) extends InsnUCode(new LdcInsnNode(x))
case class I64(x: Long) extends InsnUCode(new LdcInsnNode(x))
case class F32(x: Float) extends InsnUCode(new LdcInsnNode(x))
case class F64(x: Double) extends InsnUCode(new LdcInsnNode(x))
case class True() extends InsnUCode(new LdcInsnNode(1))
case class False() extends InsnUCode(new LdcInsnNode(0))
case class _If(cond: UCode, cnsq: UCode, altr: UCode) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) {
    val c = cond
    val lfalse = new LabelNode
    val lafter = new LabelNode
    c.emit(il)
    il += new JumpInsnNode(IFEQ, lfalse)
    cnsq.emit(il)
    il += new JumpInsnNode(GOTO, lafter)
    il += lfalse
    altr.emit(il)
    il += lafter
  }
}
case class NewInitializedArray(initializer: Array[UCode], rvb: Code[RegionValueBuilder], ti: TypeInfo[_]) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) {
    il += ti.newArray()
    initializer.zipWithIndex.foreach { case (x, i) =>
      il += new InsnNode(DUP)
      x.emit(il)
      i.emit(il)
      il += new InsnNode(ti.astoreOp)
    }
  }
}
case class ArrayRef(array: UCode, idx: UCode, ti: TypeInfo[_]) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) {
    array.emit(il)
    idx.emit(il)
    il += new InsnNode(ti.aloadOp)
  }
}
case class Erase(c: Code[_]) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) {
    c.emit(il)
  }
}
// FIXME: check something?
case class Reify[T](c: UCode) extends Code[T] {
  def emit(il: Growable[AbstractInsnNode]) {
    c.emit(il)
  }
}

case class ULocalRef(i: Int, ti: TypeInfo[_]) {
  def load(): UCode =
    new UCode {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        il += new IntInsnNode(ti.loadOp, i)
      }
    }

  def store(rhs: UCode): UCode =
    new UCode {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        rhs.emit(il)
        il += new IntInsnNode(ti.storeOp, i)
      }
    }

  def :=(rhs: UCode) = store(rhs)
}

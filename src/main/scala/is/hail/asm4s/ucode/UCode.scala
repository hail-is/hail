package is.hail.asm4s.ucode

import is.hail.asm4s._

import is.hail.annotations._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.language.existentials

trait UCode {
  def emit(il: Growable[AbstractInsnNode]): Unit
  def toConditional: Option[UCodeConditional] =
    None
  def coerceConditional: UCodeConditional =
    toConditional.getOrElse { throw new RuntimeException(s"${this} cannot be used as a conditional") }
}

abstract class InsnUCode(insn: AbstractInsnNode) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) { il += insn }
}

trait UCodeConditional extends UCode {
  override def emit(il: Growable[AbstractInsnNode]) {
    val lafter = new LabelNode
    val (ltrue, lfalse) = emitConditional(il)
    il += lfalse
    il += new LdcInsnNode(0)
    il += new JumpInsnNode(GOTO, lafter)
    il += ltrue
    il += new LdcInsnNode(1)
    il += lafter
  }

  def emitConditional(il: Growable[AbstractInsnNode]): (LabelNode, LabelNode)
}

case class Null() extends InsnUCode(new InsnNode(ACONST_NULL))
case class I32(x: Int) extends InsnUCode(new LdcInsnNode(x))
case class I64(x: Long) extends InsnUCode(new LdcInsnNode(x))
case class F32(x: Float) extends InsnUCode(new LdcInsnNode(x))
case class F64(x: Double) extends InsnUCode(new LdcInsnNode(x))
case class True() extends UCodeConditional {
  override def toConditional = Some(this)
  override def emitConditional(il: Growable[AbstractInsnNode]): (LabelNode, LabelNode) = {
    val ltrue = new LabelNode
    val lfalse = new LabelNode
    il += new JumpInsnNode(GOTO, ltrue)
    (ltrue, lfalse)
  }
}
case class False() extends UCodeConditional {
  override def toConditional = Some(this)
  override def emitConditional(il: Growable[AbstractInsnNode]): (LabelNode, LabelNode) = {
    val ltrue = new LabelNode
    val lfalse = new LabelNode
    il += new JumpInsnNode(GOTO, lfalse)
    (ltrue, lfalse)
  }
}
case class _If(cond: UCode, cnsq: UCode, altr: UCode) extends UCode {
  def emit(il: Growable[AbstractInsnNode]) {
    val c = cond.coerceConditional
    val lafter = new LabelNode
    val (ltrue, lfalse) = c.emitConditional(il)
    il += lfalse
    altr.emit(il)
    il += new JumpInsnNode(GOTO, lafter)
    il += ltrue
    cnsq.emit(il)
    // fall through
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

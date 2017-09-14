package is.hail.asm4s.ucode

import is.hail.asm4s._

import is.hail.annotations._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.language.existentials
import scala.reflect.ClassTag

trait UCode {
  def emit(fb: FunctionBuilder[_])
}

abstract class InsnUCode(insn: () => AbstractInsnNode) extends UCode {
  def emit(fb: FunctionBuilder[_]) { fb.emit(insn()) }
}
abstract class BinaryInsnUCode(one: UCode, two: UCode, insn: () => AbstractInsnNode) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    one.emit(fb)
    two.emit(fb)
    fb.emit(insn())
  }
}

case class Seq(x: Array[UCode]) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    var i = 0
    while (i < x.length) {
      x(i).emit(fb)
      i += 1
    }
  }
}
case class Null() extends InsnUCode(() => new InsnNode(ACONST_NULL))
case class I32(x: Int) extends InsnUCode(() => new LdcInsnNode(x))
case class I64(x: Long) extends InsnUCode(() => new LdcInsnNode(x))
case class F32(x: Float) extends InsnUCode(() => new LdcInsnNode(x))
case class F64(x: Double) extends InsnUCode(() => new LdcInsnNode(x))
case class True() extends InsnUCode(() => new LdcInsnNode(1))
case class False() extends InsnUCode(() => new LdcInsnNode(0))
case class If(cond: UCode, cnsq: UCode, altr: UCode) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    val lfalse = new LabelNode
    val lafter = new LabelNode
    cond.emit(fb)
    fb.emit(new JumpInsnNode(IFEQ, lfalse))
    cnsq.emit(fb)
    fb.emit(new JumpInsnNode(GOTO, lafter))
    fb.emit(lfalse)
    altr.emit(fb)
    fb.emit(lafter)
  }
}
case class Not(x: UCode) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    x.emit(fb)
    val lzero = new LabelNode
    val lafter = new LabelNode
    fb.emit(new JumpInsnNode(IFEQ, lzero))
    fb.emit(0)
    fb.emit(new JumpInsnNode(GOTO, lafter))
    fb.emit(lzero)
    fb.emit(1)
    fb.emit(lafter)
  }
}
case class And(left: UCode, right: UCode) extends BinaryInsnUCode(left, right, () => new InsnNode(IAND))
case class Or(left: UCode, right: UCode) extends BinaryInsnUCode(left, right, () => new InsnNode(IOR))
case class Let(value: UCode, typ: TypeInfo[_])(body: (UCode) => UCode) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    val x = new ULocalRef(fb.allocLocal()(typ), typ)
    x.store(value).emit(fb)
    body(x.load()).emit(fb)
  }
}
case class NewInitializedArray(initializer: Array[UCode], rvb: Code[RegionValueBuilder], ti: TypeInfo[_]) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    fb.emit(ti.newArray())
    initializer.zipWithIndex.foreach { case (x, i) =>
      fb.emit(new InsnNode(DUP))
      x.emit(fb)
      fb.emit(i)
      fb.emit(new InsnNode(ti.astoreOp))
    }
  }
}
case class ArrayRef(array: UCode, idx: UCode, ti: TypeInfo[_]) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    array.emit(fb)
    idx.emit(fb)
    fb.emit(new InsnNode(ti.aloadOp))
  }
}
case class Erase(c: Code[_]) extends UCode {
  def emit(fb: FunctionBuilder[_]) {
    c.emit(fb)
  }
}
case class Reify[T](c: UCode)(implicit tct: ClassTag[T]) extends Code[T] {
  def emit(fb: FunctionBuilder[_]) {
    c.emit(fb)
  }
}
case class ULocalRef(i: Int, ti: TypeInfo[_]) {
  def load(): UCode =
    new UCode {
      def emit(fb: FunctionBuilder[_]): Unit = {
        fb.emit(new IntInsnNode(ti.loadOp, i))
      }
    }

  def store(rhs: UCode): UCode =
    new UCode {
      def emit(fb: FunctionBuilder[_]) {
        rhs.emit(fb)
        fb.emit(new IntInsnNode(ti.storeOp, i))
      }
    }

  def :=(rhs: UCode) = store(rhs)
}

package is.hail.asm4s.ucode

import is.hail.asm4s._

import is.hail.annotations._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.language.existentials
import scala.reflect.ClassTag

trait UCode[T] {
  def emit(fb: FunctionBuilder[_]): T
  def map[U](f: T => U): UCode[U] = MappedUCode(f, this)
  def flatMap[U](f: T => UCode[U]): UCode[U] = FlatMappedUCode(f, this)
  def withFilter(p: T => Boolean): UCode[T] = FilteredUCode(p, this)
  def filter(p: T => Boolean): UCode[T] = withFilter(p)

  def >>[U](f: => UCode[U]): UCode[U] = flatMap(x => f)
  def >>=[U](f: T => UCode[U]): UCode[U] = flatMap(f)
}

case class MappedUCode[T, U](f: T => U, c: UCode[T]) extends UCode[U] {
  def emit(fb: FunctionBuilder[_]): U =
    f(c.emit(fb))
}
case class FlatMappedUCode[T, U](f: T => UCode[U], c: UCode[T]) extends UCode[U] {
  def emit(fb: FunctionBuilder[_]): U =
    f(c.emit(fb)).emit(fb)
}
case class FilteredUCode[T](p: T => Boolean, c: UCode[T]) extends UCode[T] {
  def emit(fb: FunctionBuilder[_]): T = {
    val t = c.emit(fb)
    if (p(t))
      t
    else
      throw new RuntimeException()
  }
}

abstract class InsnUCode(insn: () => AbstractInsnNode) extends UCode[Unit] {
  def emit(fb: FunctionBuilder[_]) { fb.emit(insn()) }
}

case class Empty() extends UCode[Unit] {
  def emit(fb: FunctionBuilder[_]) { }
}
case class Ret[T](t: T) extends UCode[T] {
  def emit(fb: FunctionBuilder[_]): T = t
}
case class Null() extends InsnUCode(() => new InsnNode(ACONST_NULL))
case class I32(x: Int) extends InsnUCode(() => new LdcInsnNode(x))
case class I64(x: Long) extends InsnUCode(() => new LdcInsnNode(x))
case class F32(x: Float) extends InsnUCode(() => new LdcInsnNode(x))
case class F64(x: Double) extends InsnUCode(() => new LdcInsnNode(x))
case class True() extends InsnUCode(() => new LdcInsnNode(1))
case class False() extends InsnUCode(() => new LdcInsnNode(0))
case class If[T,U,V](cond: UCode[T], cnsq: UCode[U], altr: UCode[V]) extends UCode[(T, U, V)] {
  def emit(fb: FunctionBuilder[_]): (T, U, V) = {
    val lfalse = new LabelNode
    val lafter = new LabelNode
    val t1 = cond.emit(fb)
    fb.emit(new JumpInsnNode(IFEQ, lfalse))
    val t2 = cnsq.emit(fb)
    fb.emit(new JumpInsnNode(GOTO, lafter))
    fb.emit(lfalse)
    val t3 = altr.emit(fb)
    fb.emit(lafter)
    (t1, t2, t3)
  }
}
case class Or[T,U](left: UCode[T], right: UCode[U]) extends UCode[(T, U)] {
  def emit(fb: FunctionBuilder[_]): (T, U) = {
    val tl = left.emit(fb)
    val tr = right.emit(fb)
    fb.emit(new InsnNode(IOR))
    (tl, tr)
  }
}
case class Var[T](value: UCode[T], typ: TypeInfo[_]) extends UCode[(T, UCode[_])] {
  def emit(fb: FunctionBuilder[_]): (T, UCode[_]) = {
    val x = new ULocalRef(fb.allocLocal()(typ), typ)
    val t1 = x.store(value).emit(fb)
    (t1, x.load())
  }
}
// FIXME: rewrite in terms of Var
case class Let[T, U](value: UCode[T], typ: TypeInfo[_])(body: (T, UCode[Unit]) => UCode[U]) extends UCode[U] {
  def emit(fb: FunctionBuilder[_]): U = {
    val x = new ULocalRef(fb.allocLocal()(typ), typ)
    val t1 = x.store(value).emit(fb)
    val t2 = body(t1, x.load()).emit(fb)
    t2
  }
}
case class NewInitializedArray[T: ClassTag](initializer: Array[UCode[T]], rvb: Code[RegionValueBuilder], ti: TypeInfo[_]) extends UCode[Array[T]] {
  def emit(fb: FunctionBuilder[_]): Array[T] = {
    val a = new Array[T](initializer.length)
    fb.emit(ti.newArray())
    initializer.zipWithIndex.foreach { case (x, i) =>
      fb.emit(new InsnNode(DUP))
      val t = x.emit(fb)
      a(i) = t
      fb.emit(i)
      fb.emit(new InsnNode(ti.astoreOp))
    }
    a
  }
}
case class ArrayRef[T, U](array: UCode[T], idx: UCode[U], ti: TypeInfo[_]) extends UCode[(T, U)] {
  def emit(fb: FunctionBuilder[_]): (T, U) = {
    val t1 = array.emit(fb)
    val t2 = idx.emit(fb)
    fb.emit(new InsnNode(ti.aloadOp))
    (t1, t2)
  }
}
case class Erase(c: Code[_]) extends UCode[Unit] {
  def emit(fb: FunctionBuilder[_]) {
    c.emit(fb)
  }
}
case class Reify[T](c: UCode[_])(implicit tct: ClassTag[T]) extends Code[T] {
  def emit(fb: FunctionBuilder[_]) {
    c.emit(fb)
  }
}
case class ULocalRef(i: Int, ti: TypeInfo[_]) {
  def load(): UCode[Unit] =
    new UCode[Unit] {
      def emit(fb: FunctionBuilder[_]): Unit = {
        fb.emit(new IntInsnNode(ti.loadOp, i))
      }
    }

  def store[T](rhs: UCode[T]): UCode[T] =
    new UCode[T] {
      def emit(fb: FunctionBuilder[_]): T = {
        val t = rhs.emit(fb)
        fb.emit(new IntInsnNode(ti.storeOp, i))
        t
      }
    }

  def :=[T](rhs: UCode[T]) = store(rhs)
}

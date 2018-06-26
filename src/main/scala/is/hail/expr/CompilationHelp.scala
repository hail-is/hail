package is.hail.expr

import is.hail.asm4s._
import is.hail.expr.types._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree.{AbstractInsnNode, InsnNode}

import scala.collection.generic.Growable
import scala.collection.mutable
import scala.reflect.ClassTag

// Without this, we can't use invokeStatic with CompilationHelp as a type.
case class CompilationHelp() {}

object CompilationHelp {
  def arrayToWrappedArray[T](a: Code[Array[T]])(implicit tct: ClassTag[T]): Code[IndexedSeq[T]] =
    Code.invokeStatic[CompilationHelp, Array[T], mutable.WrappedArray[T]]("__arrayToWrappedArray", a)

  def __arrayToWrappedArray(a: Array[Double]): mutable.WrappedArray[Double] = a
  def __arrayToWrappedArray(a: Array[Int]): mutable.WrappedArray[Int] = a
  def __arrayToWrappedArray(a: Array[java.lang.Integer]): mutable.WrappedArray[java.lang.Integer] = a
  def __arrayToWrappedArray(a: Array[java.lang.Long]): mutable.WrappedArray[java.lang.Long] = a
  def __arrayToWrappedArray(a: Array[java.lang.Float]): mutable.WrappedArray[java.lang.Float] = a
  def __arrayToWrappedArray(a: Array[java.lang.Double]): mutable.WrappedArray[java.lang.Double] = a
  def __arrayToWrappedArray(a: Array[java.lang.Object]): mutable.WrappedArray[java.lang.Object] = a

  def arrayOf[T](elements: IndexedSeq[Code[T]])(implicit tti: TypeInfo[T], atti: TypeInfo[Array[T]]): Code[Array[T]] = {
    val loadElements = loadElementsIntoArray(elements)
    Code(Code.newArray[T](elements.length), loadElements)
  }

  private def loadElementsIntoArray[T](elements: IndexedSeq[Code[T]])(implicit tti: TypeInfo[T]): Code[Array[T]] =
    toCodeFromIndexedSeq(elements.zipWithIndex.map { case (x, i) =>
      new Code[AnyRef] {
        def emit(il: Growable[AbstractInsnNode]) {
          il += new InsnNode(DUP)
          i.emit(il)
          x.emit(il)
          il += new InsnNode(tti.astoreOp)
        }
      }
    }).asInstanceOf[Code[Array[T]]]

}

package is.hail.asm4s.ucode

import is.hail.asm4s.TypeInfo
import is.hail.asm4s.FunctionBuilder
import scala.collection.generic.Growable
import scala.language.implicitConversions
import scala.language.existentials

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

case class UCodeM[T](action: (FunctionBuilder[_ >: Null]) => T) {
  def map[U](f: T => U): UCodeM[U] =
    UCodeM(fb => f(action(fb)))
  def flatMap[U](f: T => UCodeM[U]): UCodeM[U] =
    UCodeM(fb => f(action(fb)).action(fb))
  def withFilter(p: T => Boolean): UCodeM[T] =
    UCodeM(fb => { val t = action(fb); if (p(t)) t else throw new RuntimeException() })
  def filter(p: T => Boolean): UCodeM[T] =
    withFilter(p)

  /**
    * The user must ensure that this {@code UCodeM} refers to no more arguments
    * than {@code FunctionBuilder} {@code fb} provides.
    */
  def run[F >: Null](fb: FunctionBuilder[F])(implicit ev: T =:= Unit): F =
    delayedRun(fb)(ev)()

  def delayedRun[F >: Null](fb: FunctionBuilder[F])(implicit ev: T =:= Unit): () => F = {
    action(fb)
    fb.result()
  }

}

object UCodeM {
  def newVar(x: UCode, tti: TypeInfo[_]): UCodeM[ULocalRef] =
    UCodeM { fb =>
      val r = ULocalRef(fb.allocLocal()(tti), tti)
      fb.emit(r := x)
      r
    }

  private case class FbIsGrowable(fb: FunctionBuilder[_ >: Null]) extends Growable[AbstractInsnNode] {
    def +=(e: AbstractInsnNode) = { fb.emit(e); this }
    def clear() = throw new UnsupportedOperationException()
  }

  def mux(cond: UCode, cnsq: UCodeM[Unit], altr: UCodeM[Unit]): UCodeM[Unit] =
    UCodeM { fb =>
      val fbg = FbIsGrowable(fb)
      val c = cond.coerceConditional
      val lafter = new LabelNode
      val (ltrue, lfalse) = c.emitConditional(fbg)
      fb.emit(lfalse)
      altr.action(fb)
      fb.emit(new JumpInsnNode(GOTO, lafter))
      fb.emit(ltrue)
      cnsq.action(fb)
      fb.emit(lafter)
    }

  // def getArg(i: Int): UCodeM[UCode] =
  //   UCodeM { fb => fb.getArg(i + 1) }

  def ret[T](t: T): UCodeM[T] =
    UCodeM { fb => t }

  implicit def emit(c: UCode): UCodeM[Unit] =
    UCodeM { fb => c.emit(FbIsGrowable(fb)); () }

}

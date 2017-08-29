package is.hail.asm4s

import scala.collection.generic.Growable
import scala.language.implicitConversions
import scala.language.higherKinds

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

case class CodeM[T](action: (FunctionBuilder[_]) => T) {
  def map[U](f: T => U): CodeM[U] =
    CodeM(fb => f(action(fb)))
  def flatMap[U](f: T => CodeM[U]): CodeM[U] =
    CodeM(fb => f(action(fb)).action(fb))
  def withFilter(p: T => Boolean): CodeM[T] =
    CodeM(fb => { val t = action(fb); if (p(t)) t else throw new RuntimeException() })
  def filter(p: T => Boolean): CodeM[T] =
    withFilter(p)

  /**
    * The user must ensure that this {@code CodeM} refers to no more arguments
    * than {@code FunctionBuilder} {@code fb} provides.
    */
  def run[F >: Null](fb: FunctionBuilder[F])(implicit ev: T =:= Unit): F =
    delayedRun(fb)(ev)()

  def delayedRun[F >: Null](fb: FunctionBuilder[F])(implicit ev: T =:= Unit): () => F = {
    emit(fb)
    fb.result()
  }

  def emit(fb: FunctionBuilder[_]): T =
    action(fb)

}

object CodeM {
  def newVar[T](x: Code[T])(implicit tti: TypeInfo[T]): CodeM[LocalRef[T]] =
    CodeM { fb =>
      val r = fb.newLocal[T]
      fb.emit(r := x)
      r
    }

  def getArg[T: TypeInfo](i: Int): CodeM[Code[T]] =
    CodeM { fb => fb.getArg[T](i + 1) }

  implicit def emit[T](c: Code[T]): CodeM[Unit] =
    CodeM { fb => fb.emit(c); () }
}

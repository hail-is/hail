package is.hail.asm4s

import scala.language.implicitConversions

case class CodeWithRefs[T](action: FunctionBuilder[_] => T) {
  def map[U](f: T => U): CodeWithRefs[U] =
    CodeWithRefs(fb => f(action(fb)))
  def flatMap[U](f: T => CodeWithRefs[U]): CodeWithRefs[U] =
    CodeWithRefs(fb => f(action(fb)).action(fb))
  def withFilter(p: T => Boolean): CodeWithRefs[T] =
    CodeWithRefs(fb => { val t = action(fb); if (p(t)) t else throw new RuntimeException() })
  def filter(p: T => Boolean): CodeWithRefs[T] =
    withFilter(p)
  def run[R >: Null](fb: Function0Builder[R])(implicit ev: T <:< Code[R]): () => R =
    fb.result(action(fb))
  def run(fb: FunctionToIBuilder)(implicit ev: T <:< Code[Int]): () => Int =
    fb.result(action(fb))
}

object CodeWithRefs {
  def newVar[T](x: Code[T])(implicit tti: TypeInfo[T]): CodeWithRefs[LocalRef[T]] =
    CodeWithRefs(fb => { val r = fb.newLocal[T] ; fb.emit(r := x) ; r })

  implicit def emit[T](c: Code[T]): CodeWithRefs[Unit] =
    CodeWithRefs(fb => { fb.emit(c); () })
}

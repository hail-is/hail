package is.hail.asm4s

trait AsmFunction0[R] { def apply(): R }
trait AsmFunction1[A,R] { def apply(a: A): R }
trait AsmFunction2[A,B,R] { def apply(a: A, b: B): R }

package is.hail.asm4s

trait AsmFunction0[R] { def apply(): R }
trait AsmFunction1[A,R] { def apply(a: A): R }
trait AsmFunction2[A,B,R] { def apply(a: A, b: B): R }
trait AsmFunction3[A,B,C,R] { def apply(a: A, b: B, c: C): R }
trait AsmFunction4[A,B,C,D,R] { def apply(a: A, b: B, c: C, d: D): R }
trait AsmFunction5[A,B,C,D,E,R] { def apply(a: A, b: B, c: C, d: D, e: E): R }

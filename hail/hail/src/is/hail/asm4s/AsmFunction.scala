package is.hail.asm4s

import is.hail.annotations.Region

trait AsmFunction0[R] { def apply(): R }
trait AsmFunction1[A, R] { def apply(a: A): R }
trait AsmFunction2[A, B, R] { def apply(a: A, b: B): R }
trait AsmFunction3[A, B, C, R] { def apply(a: A, b: B, c: C): R }
trait AsmFunction4[A, B, C, D, R] { def apply(a: A, b: B, c: C, d: D): R }
trait AsmFunction5[A, B, C, D, E, R] { def apply(a: A, b: B, c: C, d: D, e: E): R }
trait AsmFunction6[A, B, C, D, E, F, R] { def apply(a: A, b: B, c: C, d: D, e: E, f: F): R }

trait AsmFunction7[A, B, C, D, E, F, G, R] {
  def apply(a: A, b: B, c: C, d: D, e: E, f: F, g: G): R
}

trait AsmFunction8[A, B, C, D, E, F, G, H, R] {
  def apply(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H): R
}

trait AsmFunction9[A, B, C, D, E, F, G, H, I, R] {
  def apply(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H, i: I): R
}

trait AsmFunction10[A, B, C, D, E, F, G, H, I, J, R] {
  def apply(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H, i: I, j: J): R
}

trait AsmFunction12[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R] {
  def apply(
    t1: T1,
    t2: T2,
    t3: T3,
    t4: T4,
    t5: T5,
    t6: T6,
    t7: T7,
    t8: T8,
    t9: T9,
    t10: T10,
    t11: T11,
    t12: T12,
  ): R
}

trait AsmFunction13[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, R] {
  def apply(
    t1: T1,
    t2: T2,
    t3: T3,
    t4: T4,
    t5: T5,
    t6: T6,
    t7: T7,
    t8: T8,
    t9: T9,
    t10: T10,
    t11: T11,
    t12: T12,
    t13: T13,
  ): R
}

trait AsmFunction1RegionUnit {
  def apply(r: Region): Unit
}

trait AsmFunction1RegionLong {
  def apply(r: Region): Long
}

trait AsmFunction2RegionLongUnit {
  def apply(r: Region, a: Long): Unit
}

trait AsmFunction2RegionLongInt {
  def apply(r: Region, a: Long): Int
}

trait AsmFunction2RegionLongLong {
  def apply(r: Region, a: Long): Long
}

trait AsmFunction3RegionLongBooleanLong {
  def apply(r: Region, a: Long, b: Boolean): Long
}

trait AsmFunction3RegionLongLongUnit {
  def apply(r: Region, a: Long, b: Long): Unit
}

trait AsmFunction3RegionLongLongBoolean {
  def apply(r: Region, a: Long, b: Long): Boolean
}

trait AsmFunction3RegionLongIntLong {
  def apply(r: Region, a: Long, b: Int): Long
}

trait AsmFunction3RegionLongLongLong {
  def apply(r: Region, a: Long, b: Long): Long
}

trait AsmFunction4RegionLongRegionLongLong {
  def apply(r1: Region, a: Long, r2: Region, b: Long): Long
}

trait AsmFunction3RegionIteratorJLongBooleanLong {
  def apply(r: Region, a: Iterator[java.lang.Long], b: Boolean): Long
}

trait AsmFunction3RegionLongIteratorJLongBoolean {
  def apply(r: Region, a: Long, b: Iterator[java.lang.Long]): Boolean
}

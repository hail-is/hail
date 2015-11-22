package com.example

// sealed
abstract class Expr
case class Const(i: Int) extends Expr
case class Symbol(s: String) extends Expr
case class Add(l: Expr, r: Expr) extends Expr
case class Neg(e: Expr) extends Expr
case class Mul(l: Expr, r: Expr) extends Expr
case class Exp(l: Expr) extends Expr

// case class: get toString, companion object with apply(), field-wise
// equality, copy, ...

val a = Symbol("a")
val b = Symbol("b")
val c = Symbol("c")

val e = Mul(a, Add(b, c))
val f = Add(Mul(a, Const(1)), Add(c, Add(Const(3), Const(5))))
val g = Add(a, Neg(a))
val j = Add(Const(2), Const(3))
val k = Exp(Const(5))

val emptyenv = Map.empty[String, Int]
val abcenv = Map("a" -> 4, "b" -> -2, "c" -> 9)

def eval1(e: Expr, env: Map[String, Int]): Int = {
  if (e.isInstanceOf[Const])
    e.asInstanceOf[Const].i
  else if (e.isInstanceOf[Add])
    eval1(e.asInstanceOf[Add].l, env) + eval1(e.asInstanceOf[Add].r, env)
  else
    throw new UnsupportedOperationException // more...
}

// scala> eval1(j, emptyenv)
// res15: Int = 5

// pattern match (more powerful switch), list of cases called
// alternatives "case PATTERN => body".  alternatives are tried in
// order.  pattern includes constants (compares values with ==),
// variable pattern (like e), wildcard pattern (_), constructor

def stupidToString(a: Any): String = a match {
  case 1 => "1"
  case "abc" => "abc"
  case 3.14 => "3.14"
  case 1 :: Nil => "List(1)"
  case _ => "Shit, dunno!"
}

def eval2(e: Expr, env: Map[String, Int]): Int = e match {
  case Const(i) => i
  case Add(l, r) => eval2(l, env) + eval2(r, env)
  case Mul(l, r) => eval2(l, env) * eval2(r, env)
  case Neg(e) => eval2(e, env)
  case Symbol(s) => env(s)
  // case _ => throw new RuntimeException("unknown expression" " + e)
}

// scala> eval2(e, abcenv)
// res13: Int = 28

// scala> eval2(k, abcenv)
// scala.MatchError: Exp(Const(5)) (of class Exp)

def simplify(e: Expr): Expr = e match {
  case Add(l, Const(0)) => l
  case Add(Const(0), r) => r
  case Add(Const(i), Const(j)) => Const(i + j)
  // case Add(e, Neg(e)) => Const(0)
  case Add(e1, Neg(e2)) if e1 == e2 => Const(0)
  case Add(l, r) => Add(simplify(l), simplify(r))

  case Mul(l, Const(1)) => l
  case Mul(Const(1), r) => r
  case Mul(Const(i), Const(j)) => Const(i * j)
  case Mul(l, r) => Mul(simplify(l), simplify(r))

  case Neg(e) => Neg(simplify(e))

  case _ => e
}

// scala> simplify(f)
// res19: Expr = Add(Symbol(a),Add(Symbol(c),Const(8)))

// scala> simplify(g)
// res20: Expr = Const(0)

val five: Int = 5

def isFive(x: Int): Boolean = x match {
  case five => true
  case _ => false
}

// lower case is pattern variable, Five, `five`

def threeSum(a: Array[Int]): Int = a match {
  case Array(a, b, c) => a + b + c
  case _ => -1
}

// scala> threeSum(Array(1, 2, 3))
// res29: Int = 6

// scala> threeSum(Array(1, 2, 3, 4))
// res30: Int = -1

def threeSum2(a: Array[Int]): Int = a match {
  case Array(a, b, c, _*) => a + b + c
  case _ => -1
}

// scala> threeSum2(Array(1, 2, 3, 4, 5))
// res35: Int = 6

def isBinaryOp(e: Expr): Boolean = e match {
  case a: Add => true
  case b: Mul => true
  case _ => false
}

// casting: e.isInstanceOf[T], e.asInstanceOf[T]

// type erasure: no type params are remembered at runtime

// scala> a.asInstanceOf[Map[String, String]]
// res42: Map[String,String] = Map(1 -> 2)

// scala> Map(1 -> 2).asInstanceOf[Map[String, String]].head._1
// java.lang.ClassCastException: java.lang.Integer cannot be cast to java.lang.String

def isIntMap(x: Any) = x match {
  case m: Map[Int, Int] => true
  case _ => false
}

// <console>:11: warning: non-variable type argument Int in type pattern scala.collection.immutable.Map[Int,Int] (the underlying of Map[Int,Int]) is unchecked since it is eliminated by erasure

// scala> isIntMap(Map("abc" -> "def"))
// res49: Boolean = true

// not true for Arrays: they are special and type parameters are
// stored.

// sealed classes

// where can you use patterns?

// val expressions:

val Add(p, q) = j

// p: Expr = Const(2)
// q: Expr = Const(3)

def sumPairs(a: Array[(Int, Int)]): Array[Int] =
  a.map { case (f, s) => f + s }

// scala> sumPairs(Array((1, 2), (3, 4), (5, 6)))
// res6: Array[Int] = Array(3, 7, 11)

// also Function.tupled, .tupled, ...

// pattern type constraints aren't used in type inference!

// bad: val f = (v, u) => v + u
// OK: val f: (Int, Int) => Int = (v, u) => v + u
// OK: val f = (v: Int, u: Int) => v + u

// WTF?  OK: val f: (Int, Int) => Int = { case (u, v) => u + v }

// bad: val f = { case (u, v) => u + v }
// bad: val f = { case (u: Int, v: Int) => u + v }
// OK: val f = { case (u, v) => u + v } : (Int, Int) => Int

// OK: val f = { case (u, v) => u + v } : ((Int, Int)) => Int

// in for loops:

def printSomes(a: Array[Option[Int]]) {
  for (Some(v) <- a)
    println(v)
}

// scala> printSomes(Array(Some(1), None, Some(2), None, Some(3)))
// 1
// 2
// 3

// this is not magic!  Extractors.

case class Locus1(chrom: String, start: Int)

object Locus2 {
  def apply(chrom: String, start: Int) =
    new Locus2(chrom, start)

  def unapply(l: Locus2): Option[(String, Int)] =
    Some((l.chrom, l.start))
}

class Locus2(val chrom: String, val start: Int) {

  override def toString(): String = s"Locus2($chrom, $start)"
}

val site = Locus2("22", 1234)

// scala> val Locus2(ch, pos) = site
// ch: String = 22
// pos: Int = 1234

// scala> val Locus2(ch, pos, other) = site
// <console>:17: error: too many patterns for object Locus2 offering (String, Int): expected 2, found 3
//       val Locus2(ch, pos, other) = site

object SuperArray {
  // a has type Seq[Int]
  def apply(a: Int*) = new SuperArray(a.toArray)

  def unapplySeq(sa: SuperArray): Option[Seq[Int]] =
    Some(sa.a.toSeq)
}

class SuperArray(val a: Array[Int]) {

}

// scala> val SuperArray(a, b, _*) = SuperArray(1, 2, 3, 4)
// a: Int = 1
// b: Int = 2

// scala> val SuperArray(a, b, _*) = SuperArray(1)
// scala.MatchError: SuperArray@1716da4 (of class SuperArray)
//   ... 33 elided

// extractors vs case classes: break dependence on representation,
// case classes are shorter, more efficient (unapply, unapplySeq can
// do anything), sealed

val email = "([^@]+)@(.*)".r

// scala> val email(name, domain) = "cseed@broadinstitute.org"
// name: String = cseed
// domain: String = broadinstitute.org

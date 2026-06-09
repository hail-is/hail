package is.hail.expr.ir

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.Memoized.eval
import is.hail.expr.ir.Scope.EVAL
import is.hail.expr.ir.defs._

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class MemoizedMapBenchmark {

  @Param(Array("5", "10", "20", "50"))
  var numBinds: Int = _

  // must not be an Atom
  private val Zero: IR = I32(0) + 0

  @Benchmark
  def memoized(): IR = {
    var x = Memoized.memo[EVAL.type](Zero)
    for (_ <- 1 until numBinds)
      x = x.map(_ + 1)
    eval(x)
  }

  @Benchmark
  def directV(): IR = {
    var x = DirectV.memo(Zero)
    for (_ <- 1 until numBinds)
      x = x.map(_ + 1)
    x.toIR
  }

  @Benchmark
  def directL(): IR = {
    var x = DirectL.memo(Zero)
    for (_ <- 1 until numBinds)
      x = x.map(_ + 1)
    x.toIR
  }

  @Benchmark
  def irbuilder(): IR =
    IRBuilder.scoped { b =>
      var x = b.strictMemoize(Zero)
      for (_ <- 1 until numBinds)
        x = b.strictMemoize(x + 1)
      x
    }
}

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class MemoizedFlatMapBenchmark {

  // motivation for `go` in benchmarks below:
  //
  //   for {
  //    a <- pure(Zero)
  //    b <- pure(a + 1)
  //    c <- pure(b + 1)
  //  } yield c + 1
  //
  //  becomes
  //
  //  pure(Zero).flatMap(a =>
  //    pure(a + 1).flatMap(b =>
  //      pure(b + 1).map(c =>
  //        c + 1)))

  @Param(Array("5", "10", "20", "50"))
  var numBinds: Int = _

  // must not be an Atom
  private val Zero: IR = I32(0) + 0

  @Benchmark
  def memoized(): IR = {
    def go(n: Int): Memoized[EVAL.type] =
      if (n <= 0) Memoized.memo(Zero)
      else Memoized.memo(Zero).flatMap(acc => go(n - 1).map(acc + _))

    eval(go(numBinds))
  }

  @Benchmark
  def directV(): IR = {
    def go(n: Int): DirectV =
      if (n <= 0) DirectV.memo(Zero)
      else DirectV.memo(Zero).flatMap(acc => go(n - 1).map(acc + _))
    go(numBinds).toIR
  }

  @Benchmark
  def directL(): IR = {
    def go(n: Int): DirectL =
      if (n <= 0) DirectL.memo(Zero)
      else DirectL.memo(Zero).flatMap(acc => go(n - 1).map(acc + _))
    go(numBinds).toIR
  }

  @Benchmark
  def irbuilder(): IR =
    IRBuilder.scoped { b =>
      def go(n: Int): IR =
        if (n <= 0) b.strictMemoize(Zero)
        else b.strictMemoize(Zero) + go(n - 1)

      go(numBinds)
    }
}

final class DirectV(val bindings: Vector[Binding], val body: IR) {
  def map(f: Atom => IR): DirectV = {
    val ref = Ref(freshName(), body.typ)
    new DirectV(bindings :+ Binding(ref.name, body, EVAL), f(ref))
  }

  def flatMap(f: Atom => DirectV): DirectV = {
    val ref = Ref(freshName(), body.typ)
    val res = f(ref)
    new DirectV(
      (bindings :+ Binding(ref.name, body, EVAL)) ++ res.bindings,
      res.body,
    )
  }

  def toIR: IR = new Block(bindings, body)
}

object DirectV {
  def memo(ir: IR): DirectV =
    new DirectV(Vector(), ir)
}

final class DirectL(val rbinds: List[Binding], val body: IR) {
  def map(f: Atom => IR): DirectL = {
    val ref = Ref(freshName(), body.typ)
    new DirectL(Binding(ref.name, body, EVAL) :: rbinds, f(ref))
  }

  def flatMap(f: Atom => DirectL): DirectL = {
    val ref = Ref(freshName(), body.typ)
    val that = f(ref)
    new DirectL(that.rbinds ::: Binding(ref.name, body, EVAL) :: rbinds, that.body)
  }

  def toIR: IR = Block(rbinds.reverseIterator.to(ArraySeq), body)
}

object DirectL {
  def memo(ir: IR): DirectL =
    new DirectL(Nil, ir)
}

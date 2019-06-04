package is.hail.backend.local

import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.ir._
import is.hail.utils.{ExecutionTimer, Timings}

import scala.reflect.ClassTag

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T]

object LocalBackend extends Backend {

  def broadcast[T: ClassTag](value: T): LocalBroadcastValue[T] = new LocalBroadcastValue(value)

  def parallelizeAndComputeWithIndex[T : ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    collection.zipWithIndex.map { case (elt, i) => f(elt, i) }.toArray
  }

  def execute(ir0: IR, optimize: Boolean = true): (Any, Timings) = {
    val timer = new ExecutionTimer("Just Interpret")
    var ir = ir0

    println(("LocalBackend.execute got", Pretty(ir)))

    ir = ir.unwrap
    if (optimize)
      ir = timer.time(
        Optimize(ir, noisy = true, canGenerateLiterals = true, context = Some(s"LocalBackend.execute - first pass")),
        "optimize first pass")
    ir = timer.time(LowerMatrixIR(ir), "lower MatrixIR")
    if (optimize)
      ir = timer.time(
        Optimize(ir, noisy = true, canGenerateLiterals = false, context = Some("LocalBackend.execute - after MatrixIR lowering")),
        "optimize after matrix lowering")
    ir = timer.time(LiftNonCompilable(EvaluateRelationalLets(ir)).asInstanceOf[IR], "lifting non-compilable")

    println(("LocalBackend.execute to lower", Pretty(ir)))

    ir = timer.time(LowerTableIR.lower(ir), "lowering TableIR")

    println(("LocalBackend.execute lowered", Pretty(ir)))

    if (optimize)
      ir = timer.time(
        Optimize(ir, noisy = true, canGenerateLiterals = false, context = Some("LocalBackend.execute - after TableIR lowering")),
        "optimize after table lowering")

    println(("LocalBackend.execute", Pretty(ir)))

    val value = timer.time(Interpret[Any](ir), "runtime")
    (value, timer.timings)
  }
}

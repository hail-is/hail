package is.hail.backend.spark

import is.hail.{HailContext, cxx}
import is.hail.annotations.{Region, SafeRow}
import is.hail.backend.{Backend, BroadcastValue, LowerTableIR, LowererUnsupportedOperation}
import is.hail.cxx.CXXUnsupportedOperation
import is.hail.expr.ir._
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual.TVoid
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

object SparkBackend {
  def executeJSON(ir: IR): String = HailContext.backend.executeJSON(ir)
}

class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

case class SparkBackend(sc: SparkContext) extends Backend {

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new SparkBroadcastValue[T](sc.broadcast(value))

  def parallelizeAndComputeWithIndex[T : ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val rdd = sc.parallelize[T](collection, numSlices = collection.length)
    rdd.mapPartitionsWithIndex { (i, it) =>
      val elt = it.next()
      assert(!it.hasNext)
      Iterator.single(f(elt, i))
    }.collect()
  }

  override def cxxLowerAndExecute(ir0: IR, optimize: Boolean = true): (Any, Timings) = {
    val timer = new ExecutionTimer("Backend.execute")
    val ir = lower(ir0, Some(timer), optimize)

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ir)}")

    val res = ir.typ match {
      case TVoid =>
        val f = timer.time(cxx.Compile(ir, optimize), "CXX compile")
        timer.time(Region.scoped { region => f(region.get()) }, "Runtime")
        Unit
      case _ =>
        val pipeline = MakeTuple(FastIndexedSeq(ir))
        val f = timer.time(cxx.Compile(pipeline, optimize: Boolean), "CXX compile")
        timer.time(
          Region.scoped { region =>
            val off = f(region.get())
            SafeRow(pipeline.pType.asInstanceOf[PTuple], region, off).get(0)
          },
          "Runtime")
    }

    (res, timer.timings)
  }

  override def asSpark(): SparkBackend = this
}

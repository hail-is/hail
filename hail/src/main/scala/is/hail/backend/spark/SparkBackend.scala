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

  def jvmLowerAndExecute(ir0: IR, optimize: Boolean = true): (Any, Timings) = {
    val timer = new ExecutionTimer("JVMCompile")

    val ir = LowerTableIR(ir0, Some(timer), optimize)

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ir)}")

    val v = Region.scoped { region =>
      ir.typ match {
        case TVoid =>
          val (_, f) = timer.time(Compile[Unit](ir), "SparkBackend.execute - JVM compile")
          timer.time(f(0)(region), "SparkBackend.execute - Runtime")
        case _ =>
          println(Pretty(ir))
          val (pt: PTuple, f) = timer.time(Compile[Long](MakeTuple(FastSeq(ir))), "SparkBackend.execute - JVM compile")
          timer.time(SafeRow(pt, region, f(0)(region)).get(0), "SparkBackend.execute - Runtime")
      }
    }

    (v, timer.timings)
  }

  def cxxExecute(ir0: IR, optimize: Boolean = true): (Any, Timings) = {
    val timer = new ExecutionTimer("CXX Compile")

    val ir = try {
      LowerTableIR(ir0, Some(timer), optimize)
    } catch {
      case e: LowererUnsupportedOperation =>
        throw new CXXUnsupportedOperation(s"Failed lowering step:\n${e.getMessage}")
    }

    val value = ir.typ match {
      case TVoid =>
        val f = timer.time(cxx.Compile(ir, optimize), "SparkBackend.execute - CXX compile")
        timer.time(Region.scoped { region => f(region.get()) }, "SparkBackend.execute - Runtime")
        Unit
      case _ =>
        val pipeline = MakeTuple(FastIndexedSeq(ir))
        val f = timer.time(cxx.Compile(pipeline, optimize: Boolean), "SparkBackend.execute - CXX compile")
        timer.time(
          Region.scoped { region =>
            val off = f(region.get())
            SafeRow(pipeline.pType.asInstanceOf[PTuple], region, off).get(0)
          },
          "SparkBackend.execute - Runtime")
    }

    (value, timer.timings)
  }

  def execute(ir: IR, optimize: Boolean = true): (Any, Timings) = {
    try {
      if (HailContext.get.flags.get("cpp") == null)
        jvmLowerAndExecute(ir, optimize)
      else
        cxxExecute(ir, optimize)
    } catch {
      case (_: CXXUnsupportedOperation | _: LowererUnsupportedOperation) =>
        CompileAndEvaluate(ir, optimize = optimize)
//      case e: Throwable =>
//        println(Pretty(ir))
//        CompileAndEvaluate(ir, optimize = optimize)
    }
  }

  override def asSpark(): SparkBackend = this
}

package is.hail.backend.service

import is.hail.annotations.UnsafeRow
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LoweringPipeline}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, MakeTuple}
import is.hail.expr.types.physical.PBaseStruct
import is.hail.io.fs.GoogleStorageFS
import is.hail.utils.FastIndexedSeq
import org.json4s.JsonAST.{JObject, JString}
import org.json4s.jackson.JsonMethods

import scala.collection.mutable
import scala.reflect.ClassTag

object ServiceBackend {
  def apply(): ServiceBackend = {
    new ServiceBackend()
  }
}

class User(
  val username: String,
  val fs: GoogleStorageFS)

class ServiceBackend() extends Backend {
  private[this] val users = mutable.Map[String, User]()

  def addUser(username: String, key: String): Unit = {
    assert(!users.contains(username))
    users += username -> new User(username, new GoogleStorageFS(key))
  }

  def removeUser(username: String): Unit = {
    assert(users.contains(username))
    users -= username
  }

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val n = collection.length
    val r = new Array[U](n)
    var i = 0
    while (i < n) {
      r(i) = f(collection(i), i)
      i += 1
    }
    r
  }

  def stop(): Unit = ()

  def valueType(username: String, s: String): String = {
    val x = IRParser.parse_value_ir(s)
    x.typ.toString
  }

  def tableType(username: String, s: String): String = {
    val x = IRParser.parse_table_ir(s)
    x.typ.toString
  }

  def matrixTableType(username: String, s: String): String = {
    val x = IRParser.parse_matrix_ir(s)
    x.typ.toString
  }

  def blockMatrixType(username: String, s: String): String = {
    val x = IRParser.parse_blockmatrix_ir(s)
    x.typ.toString
  }

  def execute(username: String, s: String): String = {
    val user = users(username)
    ExecuteContext.scoped(this, user.fs) { ctx =>
      var x = IRParser.parse_value_ir(s)
      x = LoweringPipeline.darrayLowerer(DArrayLowering.All).apply(ctx, x, optimize = true)
        .asInstanceOf[IR]
      val (pt, f) = Compile[Long](ctx, MakeTuple.ordered(FastIndexedSeq(x)), None, optimize = true)

      val a = f(0, ctx.r)(ctx.r)
      val v = new UnsafeRow(pt.asInstanceOf[PBaseStruct], ctx.r, a)

      JsonMethods.compact(
        JObject(List("value" -> JSONAnnotationImpex.exportAnnotation(v.get(0), x.typ),
          "type" -> JString(pt.virtualType.toString))))
    }
  }
}

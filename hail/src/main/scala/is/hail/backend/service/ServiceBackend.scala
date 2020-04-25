package is.hail.backend.service

import java.io.{PrintWriter, StringWriter}

import is.hail.annotations.{Region, UnsafeRow}
import is.hail.asm4s._
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LowererUnsupportedOperation, LoweringPipeline}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, MakeTuple}
import is.hail.expr.types.physical.{PBaseStruct, PType}
import is.hail.io.fs.GoogleStorageFS
import is.hail.utils._
import org.json4s.JsonAST.{JArray, JBool, JInt, JObject, JString}
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
  val tmpdir: String,
  val fs: GoogleStorageFS)

final class Response(val status: Int, val value: String)

class ServiceBackend() extends Backend {
  private[this] val users = mutable.Map[String, User]()

  def addUser(username: String, key: String): Unit = {
    assert(!users.contains(username))
    // FIXME
    users += username -> new User(username, "/tmp", new GoogleStorageFS(key))
  }

  def removeUser(username: String): Unit = {
    assert(users.contains(username))
    users -= username
  }

  def userContext[T](username: String)(f: (ExecuteContext) => T): T = {
    val user = users(username)
    ExecuteContext.scoped(user.tmpdir, "file:///tmp", this, user.fs)(f)
  }

  def defaultParallelism: Int = 10

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

  def statusForException(f: => String): Response = {
    try {
      new Response(200, f)
    } catch {
      case e: HailException =>
        new Response(400, e.getMessage)
      case e: LowererUnsupportedOperation =>
        new Response(400, e.getMessage)
      case e: Exception =>
        using(new PrintWriter(new StringWriter())) { pw =>
          e.printStackTrace(pw)
          new Response(500, pw.toString)
        }
    }
  }

  def valueType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        val x = IRParser.parse_value_ir(ctx, s)
        x.typ.toString
      }
    }
  }

  def tableType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        val x = IRParser.parse_table_ir(ctx, s)
        val t = x.typ
        val jv = JObject("global" -> JString(t.globalType.toString),
          "row" -> JString(t.rowType.toString),
          "row_key" -> JArray(t.key.map(f => JString(f)).toList))
        JsonMethods.compact(jv)
      }
    }
  }

  def matrixTableType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        val x = IRParser.parse_matrix_ir(ctx, s)
        val t = x.typ
        val jv = JObject("global" -> JString(t.globalType.toString),
          "col" -> JString(t.colType.toString),
          "col_key" -> JArray(t.colKey.map(f => JString(f)).toList),
          "row" -> JString(t.rowType.toString),
          "row_key" -> JArray(t.rowKey.map(f => JString(f)).toList),
          "entry" -> JString(t.entryType.toString))
        JsonMethods.compact(jv)
      }
    }
  }

  def blockMatrixType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        val x = IRParser.parse_blockmatrix_ir(ctx, s)
        val t = x.typ
        val jv = JObject("element_type" -> JString(t.elementType.toString),
          "shape" -> JArray(t.shape.map(s => JInt(s)).toList),
          "is_row_vector" -> JBool(t.isRowVector),
          "block_size" -> JInt(t.blockSize))
        JsonMethods.compact(jv)
      }
    }
  }

  def execute(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        var x = IRParser.parse_value_ir(ctx, s)
        x = LoweringPipeline.darrayLowerer(DArrayLowering.All).apply(ctx, x, optimize = true)
          .asInstanceOf[IR]
        val (pt, f) = Compile[AsmFunction1RegionLong](ctx,
          FastIndexedSeq[(String, PType)](),
          FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
          MakeTuple.ordered(FastIndexedSeq(x)),
          optimize = true)

        val a = f(0, ctx.r)(ctx.r)
        val v = new UnsafeRow(pt.asInstanceOf[PBaseStruct], ctx.r, a)

        JsonMethods.compact(
          JObject(List("value" -> JSONAnnotationImpex.exportAnnotation(v.get(0), x.typ),
            "type" -> JString(x.typ.toString))))
      }
    }
  }
}

package is.hail.backend

import is.hail.expr.ir.{BaseIR, BlockMatrixIR, ExecuteContext, IR, IRParser, IRParserEnvironment, MatrixIR, TableIR}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.types.encoded.EType
import org.json4s.jackson.JsonMethods

import scala.collection.mutable
import scala.collection.JavaConverters._

abstract class Py4JBackend extends Backend {
  var irMap: mutable.Map[Long, BaseIR] = new mutable.HashMap[Long, BaseIR]()

  private[this] var irCounter: Long = 0

  def addIR(x: BaseIR): Long = {
    val id = irCounter
    irCounter += 1
    irMap(id) = x
    id
  }

  def removeIR(id: Long): Unit = {
    irMap -= id
  }

  def withExecuteContext[T]()(f: ExecuteContext => T): T

  def pyParseValueIR(s: String, refMap: java.util.Map[String, String]): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_value_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyParseTableIR(s: String, refMap: java.util.Map[String, String]): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_table_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyParseMatrixIR(s: String, refMap: java.util.Map[String, String]): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyParseBlockMatrixIR(
    s: String, refMap: java.util.Map[String, String]
  ): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyValueType(id: Long): String = {
    irMap(id).typ.toString
  }

  def pyTableType(id: Long): String = {
    JsonMethods.compact(
      irMap(id).asInstanceOf[TableIR].typ.toJSON)
  }

  def pyMatrixType(id: Long): String = {
    JsonMethods.compact(
      irMap(id).asInstanceOf[MatrixIR].typ.toJSON)
  }

  def pyBlockMatrixType(id: Long): String = {
    JsonMethods.compact(
      irMap(id).asInstanceOf[BlockMatrixIR].typ.toJSON)
  }

  def executeJSON(id: Long): String

  override def stop(): Unit = {
    irMap = null
  }

  def pyBlockMatrixIsSparse(id: Int): Boolean = {
    irMap(id).asInstanceOf[BlockMatrixIR].typ.isSparse
  }
}

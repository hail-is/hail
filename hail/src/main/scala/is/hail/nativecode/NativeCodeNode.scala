package is.hail.nativecode

import is.hail.utils.ArrayBuilder

abstract class NativeCodeStatement extends Serializable {

  def toString: String

}

class NativeCodeVariable(name: String, t: String, init: NativeCodeStatement = null) extends Serializable {

  def declare: String = s"$t $name${if (init == null) "" else s" = $init"};\n"

}

object NCode {
  def apply(expr: String): NativeCodeStatement = new NativeCodeStatement {
      override def toString: String = expr
  }
}

trait NativeScope {

  private val refs: ArrayBuilder[NativeCodeVariable] = new ArrayBuilder[NativeCodeVariable]()

  private val statements: ArrayBuilder[NativeCodeStatement] = new ArrayBuilder[NativeCodeStatement]()

  def newRef(name: String, t: String = "auto", init: NativeCodeStatement = null): NativeCodeVariable = {
    val r = new NativeCodeVariable(name, t, init)
    refs += r
    r
  }

  def emit(statement: NativeCodeStatement): Unit = {
    statements += statement
  }

  override def toString: String = {
    val sb = new StringBuilder()
    sb.append("{\n")
    refs.result().foreach(ref => sb.append(ref.declare))
    statements.result().foreach(statement => sb.append(s"$statement;\n"))
    sb.append("}")
    sb.result()
  }
}

class NativeFunction(name: String, resultType: String, args: Array[(String, String)]) extends NativeScope with Serializable {

  def signature: String = s"$resultType $name(${args.map { case (n, t) => s"$t $n" } .mkString(", ")})"

  override def toString: String = s"$signature ${super.toString}"
}

class NativeFile extends Serializable {

  val includes: ArrayBuilder[String] = new ArrayBuilder[String]()

  val functions: ArrayBuilder[NativeFunction] = new ArrayBuilder[NativeFunction]()

  val makeObjectHolder = new NativeFunction("makeObjectHolder", "NativeObjPtr", Array("st" -> "NativeStatus *", "objects" -> "long"))
  makeObjectHolder.emit(NCode("return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(objects))"))
  functions += makeObjectHolder

  def include(include: String): Unit = {
    includes += include
  }

  def addFunction(name: String, resultType: String, args: Array[(String, String)]): NativeFunction = {
    val f = new NativeFunction(name, resultType, args)
    functions += f
    f
  }

  def build(name: String): (Array[java.lang.Object] => Long) = {
    val sb = new StringBuilder()
    includes.result().foreach { inc => sb.append(s"#include $inc\n") }

    sb.append(
      s"""
         |NAMESPACE_HAIL_MODULE_BEGIN
         |
         |class ObjectHolder : public NativeObj {
         | public:
         |  ObjectArrayPtr objects_;
         |
         |  ObjectHolder(ObjectArray* objects) :
         |    objects_(std::dynamic_pointer_cast<ObjectArray>(objects->shared_from_this())) {
         |  }
         |};
         |
         |""".stripMargin)

    functions.result().foreach(f => sb.append(s"$f\n\n"))
    sb.append("NAMESPACE_HAIL_MODULE_END\n")

    val function = sb.toString()
    println(function)

    { objects: Array[java.lang.Object] =>
      val st = new NativeStatus()
      val mod = new NativeModule("", function)
      val makeObjectHolder = mod.findPtrFuncL1(st, "makeObjectHolder")
      val f = mod.findLongFuncL1(st, name)
      mod.close()
      val objArray = new ObjectArray(objects)
      val holder = new NativePtr(makeObjectHolder, st, objArray.get())
      objArray.close()
      val result = f(st, holder.get())
      f.close()
      result
    }
  }

}
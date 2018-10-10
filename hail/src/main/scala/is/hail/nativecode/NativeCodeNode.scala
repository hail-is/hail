package is.hail.nativecode

import is.hail.utils.ArrayBuilder

trait NativeCodeUnit {
  def toString: String
}

abstract class NativeCodeStatement extends NativeCodeUnit {

  def toString: String

}

class NativeCodeVariable(name: String, t: String, init: NativeCodeStatement = null) extends Serializable {

  def declare: String = s"$t $name${if (init == null) "" else s" = $init"};\n"

  def load(): NativeCodeStatement = NCode(name)

  def store(v: NativeCodeStatement): NativeCodeStatement = NCode(s"$name = $v")

  override def toString: String = name
}

object NCode {
  def apply(expr: String): NativeCodeStatement = new NativeCodeStatement {
      override def toString: String = expr
  }
}

trait NativeScope extends NativeCodeUnit {

  private val refs: ArrayBuilder[NativeCodeVariable] = new ArrayBuilder[NativeCodeVariable]()

  private val statements: ArrayBuilder[NativeCodeUnit] = new ArrayBuilder[NativeCodeUnit]()

  def depth: Int

  def newRef(name: String, t: String = "auto", init: NativeCodeStatement = null): NativeCodeVariable = {
    val r = new NativeCodeVariable(s"field${refs.size}_d${depth}_$name", t, init)
    refs += r
    r
  }

  def emit(statement: NativeCodeUnit): Unit = {
    statements += statement
  }

  def doWhile(while_statement: NativeCodeStatement): NativeDoWhile = {
    val scope = new NativeDoWhile(this, while_statement)
    emit(scope)
    scope
  }

  def _while(while_statement: NativeCodeStatement): NativeWhile = {
    val scope = new NativeWhile(this, while_statement)
    emit(scope)
    scope
  }

  def _if(if_statement: NativeCodeStatement): NativeIf = {
    val scope = new NativeIf(this, if_statement, None)
    emit(scope)
    scope
  }

  def _if(if_statement: NativeCodeStatement, else_block: NativeScope): NativeIf = {
    val scope = new NativeIf(this, if_statement, Some(else_block))
    emit(scope)
    scope
  }

  override def toString: String = {
    val sb = new StringBuilder()
    sb.append("{\n")
    refs.result().foreach(ref => sb.append(ref.declare))
    statements.result().foreach(statement => statement match {
      case _: NativeCodeStatement => sb.append(s"$statement;\n")
      case _ => sb.append(s"$statement")
    })
    sb.append("}")
    sb.result()
  }
}

class NativeDoWhile(parent: NativeScope, while_statement: NativeCodeStatement) extends NativeScope {

  val depth: Int = parent.depth + 1

  override def toString: String =
    s"do ${super.toString()} while (${while_statement.toString});\n"
}

class NativeWhile(parent: NativeScope, while_statement: NativeCodeStatement) extends NativeScope {

  val depth: Int = parent.depth + 1

  override def toString: String =
    s"while (${while_statement.toString}) ${super.toString()}\n"
}

class NativeIf(parent: NativeScope, cond: NativeCodeStatement, else_block: Option[NativeScope] = None) extends NativeScope {

  val depth: Int = parent.depth + 1

  override def toString: String =
    s"if (${cond.toString}) ${super.toString}${else_block.map(b => s" else $b").getOrElse("")}\n"
}

class NativeFunction(val file: NativeFile, val name: String, resultType: String, argSig: Array[(String, String)]) extends NativeScope with Serializable {

  val depth: Int = 0

  def signature: String = s"$resultType $name(${argSig.map { case (n, t) => s"$t $n" } .mkString(", ")})"

  def call(args: NativeCodeStatement*): NativeCodeStatement = {
    assert(args.length == argSig.length)
    NCode(s"$name(${args.mkString(", ")})")
  }

  def getArg(i: Int): NativeCodeStatement = NCode(argSig(i)._1)

  override def toString: String = s"$signature ${super.toString}"
}

class NativeFile extends Serializable {

  val includes: ArrayBuilder[String] = new ArrayBuilder[String]()

  val functions: ArrayBuilder[NativeFunction] = new ArrayBuilder[NativeFunction]()

  val makeObjectHolder = new NativeFunction(this, "makeObjectHolder", "NativeObjPtr", Array("st" -> "NativeStatus *", "objects" -> "long"))
  makeObjectHolder.emit(NCode("return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(objects))"))
  functions += makeObjectHolder

  def include(include: String): Unit = {
    includes += include
  }

  def addFunction(name: String, resultType: String, args: Array[(String, String)]): NativeFunction = {
    val f = new NativeFunction(this, s"function${functions.size}_$name", resultType, args)
    functions += f
    f
  }

  def build(name: String, options: String = "-O1"): () => (Array[java.lang.Object] => Long) = {
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

    val code = new PrettyCode(sb.toString()).toString()

    println(code)

    val mod = new NativeModule(options, code)
    val st = new NativeStatus()
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    st.close()
    val modKey = mod.getKey()
    val modBinary = mod.getBinary()
    mod.close()

    new (() => ((Array[java.lang.Object]) => Long)) with java.io.Serializable {

      def apply(): (Array[java.lang.Object]) => Long = { args: Array[java.lang.Object] =>
        val st = new NativeStatus()
        val mod = new NativeModule(modKey, modBinary)
        mod.findOrBuild(st)
        assert(st.ok, st.toString())
        val makeObjectHolder = mod.findPtrFuncL1(st, "makeObjectHolder")
        assert(st.ok, st.toString())
        val f = mod.findLongFuncL1(st, name)
        assert(st.ok, s"$name: ${st.toString()}")
        mod.close()
        val objArray = new ObjectArray(args)
        val holder = new NativePtr(makeObjectHolder, st, objArray.get())
        objArray.close()
        val result = f(st, holder.get())
        f.close()
        result
      }
    }
  }

}
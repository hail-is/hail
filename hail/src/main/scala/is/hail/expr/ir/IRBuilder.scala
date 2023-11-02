package is.hail.expr.ir

import is.hail.utils.BoxedArrayBuilder

object IRBuilder {
  def scoped(f: IRBuilder => IR): IR = {
    val builder = new IRBuilder()
    val result = f(builder)
    Let(builder.bindings.underlying(), result)
  }
}

class IRBuilder() {
  private val bindings: BoxedArrayBuilder[(String, IR)] =
    new BoxedArrayBuilder[(String, IR)]()

  def getBindings: IndexedSeq[(String, IR)] = bindings.result()

  def memoize(ir: IR): TrivialIR = ir match {
    case ir: TrivialIR => ir
    case _ => strictMemoize(ir)
  }

  def strictMemoize(ir: IR): Ref = {
    if (!ir.typ.isRealizable)
      throw new RuntimeException(s"IR ${ir.getClass.getName} of type ${ir.typ} is not realizable")
    val name = genUID()
    bindings += name -> ir
    Ref(name, ir.typ)
  }
}

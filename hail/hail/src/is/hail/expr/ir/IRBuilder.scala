package is.hail.expr.ir

import is.hail.expr.ir.defs.{Let, Ref, TrivialIR}
import is.hail.utils.compat.immutable.ArraySeq

object IRBuilder {
  def scoped(f: IRBuilder => IR): IR = {
    val builder = new IRBuilder()
    val result = f(builder)
    Let(builder.getBindings, result)
  }
}

class IRBuilder {
  private val bindings = ArraySeq.newBuilder[(Name, IR)]

  def getBindings: IndexedSeq[(Name, IR)] = bindings.result()

  def memoize(ir: IR): TrivialIR = ir match {
    case ir: TrivialIR => ir
    case _ => strictMemoize(ir)
  }

  def strictMemoize(ir: IR): Ref =
    strictMemoize(ir, freshName())

  def strictMemoize(ir: IR, name: Name): Ref = {
    bindings += name -> ir
    Ref(name, ir.typ)
  }
}

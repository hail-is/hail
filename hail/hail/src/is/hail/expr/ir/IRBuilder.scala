package is.hail.expr.ir

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.defs.{Atom, Let, Ref}

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

  def memoize(ir: IR): Atom = ir match {
    case ir: Atom => ir
    case _ => strictMemoize(ir)
  }

  def strictMemoize(ir: IR): Atom =
    strictMemoize(ir, freshName())

  def strictMemoize(ir: IR, name: Name): Atom = {
    bindings += name -> ir
    Ref(name, ir.typ)
  }
}

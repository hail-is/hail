package is.hail.expr.ir

import scala.collection.mutable

object IRBuilder {
  def scoped(f: IRBuilder => IR): IR = {
    val ctx = new IRBuilder()
    val result = f(ctx)
    ctx.wrap(result)
  }
}

class IRBuilder() {
  private val bindings: mutable.ArrayBuffer[(String, IR)] = mutable.ArrayBuffer()

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

  def wrap(ir: IR): IR = {
    bindings.foldRight[IR](ir) { case ((f, v), accum) => Let(f, v, accum) }
  }
}

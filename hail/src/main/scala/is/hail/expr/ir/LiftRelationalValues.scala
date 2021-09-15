package is.hail.expr.ir

import is.hail.types.virtual.TVoid
import is.hail.utils.BoxedArrayBuilder

object LiftRelationalValues {

  def apply(ir0: IR): IR = {

    def rewrite(ir: BaseIR, ab: BoxedArrayBuilder[(String, IR)]): BaseIR = ir match {
      case RelationalLet(name, value, body) =>
        val value2 = rewrite(value, ab).asInstanceOf[IR]
        val ab2 = new BoxedArrayBuilder[(String, IR)]
        val body2 = rewrite(body, ab2).asInstanceOf[IR]
        RelationalLet(name, value2, ab2.result().foldRight[IR](body2) { case ((name, value), acc) => RelationalLet(name, value, acc) })
      case LiftMeOut(child) =>
        val ref = RelationalRef(genUID(), child.typ)
        val newChild = rewrite(child, ab).asInstanceOf[IR]
        ab += ((ref.name, newChild))
        ref
      case (_: TableAggregate
           | _: TableCount
           | _: TableToValueApply
           | _: BlockMatrixToValueApply
           | _: TableCollect
           | _: BlockMatrixCollect
           | _: TableGetGlobals) if ir.typ != TVoid =>
        val ref = RelationalRef(genUID(), ir.asInstanceOf[IR].typ)
        val rwChildren = ir.children.map(rewrite(_, ab))
        val newChild = if ((rwChildren, ir.children).zipped.forall(_ eq _))
          ir
        else
          ir.copy(rwChildren)
        ab += ((ref.name, newChild.asInstanceOf[IR]))
        ref
      case x =>
        val rwChildren = x.children.map(rewrite(_, ab))
        if ((rwChildren, ir.children).zipped.forall(_ eq _))
          ir
        else
          ir.copy(rwChildren)
    }

    val ab = new BoxedArrayBuilder[(String, IR)]
    val rw = rewrite(ir0, ab).asInstanceOf[IR]
    ab.result().foldRight[IR](rw) { case ((name, value), acc) => RelationalLet(name, value, acc)
    }
  }
}

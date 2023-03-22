package is.hail.expr.ir

import is.hail.types.virtual.TVoid
import is.hail.utils.BoxedArrayBuilder

import scala.collection.mutable

object LiftRelationalValues {

  def apply(ir0: BaseIR): BaseIR = {

    def rewrite(ir: BaseIR, ab: BoxedArrayBuilder[(String, IR)], memo: mutable.Map[IR, String]): BaseIR = ir match {
      case RelationalLet(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        val ab2 = new BoxedArrayBuilder[(String, IR)]
        val memo2 = mutable.Map.empty[IR, String]
        val body2 = rewrite(body, ab2, memo2).asInstanceOf[IR]
        RelationalLet(name, value2, ab2.result().foldRight[IR](body2) { case ((name, value), acc) => RelationalLet(name, value, acc) })
      case RelationalLetTable(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        val ab2 = new BoxedArrayBuilder[(String, IR)]
        val memo2 = mutable.Map.empty[IR, String]
        val body2 = rewrite(body, ab2, memo2).asInstanceOf[TableIR]
        RelationalLetTable(name, value2, ab2.result().foldRight[TableIR](body2) { case ((name, value), acc) => RelationalLetTable(name, value, acc) })
      case RelationalLetMatrixTable(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        val ab2 = new BoxedArrayBuilder[(String, IR)]
        val memo2 = mutable.Map.empty[IR, String]
        val body2 = rewrite(body, ab2, memo2).asInstanceOf[MatrixIR]
        RelationalLetMatrixTable(name, value2, ab2.result().foldRight[MatrixIR](body2) { case ((name, value), acc) => RelationalLetMatrixTable(name, value, acc) })
      case RelationalLetBlockMatrix(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        val ab2 = new BoxedArrayBuilder[(String, IR)]
        val memo2 = mutable.Map.empty[IR, String]
        val body2 = rewrite(body, ab2, memo2).asInstanceOf[BlockMatrixIR]
        RelationalLetBlockMatrix(name, value2, ab2.result().foldRight[BlockMatrixIR](body2) { case ((name, value), acc) => RelationalLetBlockMatrix(name, value, acc) })
      case LiftMeOut(child) =>
        val name = memo.get(child) match {
          case Some(name) => name
          case None =>
            val name = genUID()
            val newChild = rewrite(child, ab, memo).asInstanceOf[IR]
            ab += ((name, newChild))
            memo(child) = name
            name
        }
        RelationalRef(name, child.typ)
      case (_: TableAggregate
            | _: TableCount
            | _: TableToValueApply
            | _: BlockMatrixToValueApply
            | _: TableCollect
            | _: BlockMatrixCollect
            | _: TableGetGlobals) if ir.typ != TVoid =>
        val ref = RelationalRef(genUID(), ir.asInstanceOf[IR].typ)
        val rwChildren = ir.children.map(rewrite(_, ab, memo))
        val newChild = if ((rwChildren, ir.children).zipped.forall(_ eq _))
          ir
        else
          ir.copy(rwChildren)
        ab += ((ref.name, newChild.asInstanceOf[IR]))
        ref
      case x =>
        val rwChildren = x.children.map(rewrite(_, ab, memo))
        if ((rwChildren, ir.children).zipped.forall(_ eq _))
          ir
        else
          ir.copy(rwChildren)
    }

    val ab = new BoxedArrayBuilder[(String, IR)]
    val memo = mutable.Map.empty[IR, String]
    rewrite(ir0, ab, memo) match {
      case rw: IR => ab.result().foldRight[IR](rw) { case ((name, value), acc) => RelationalLet(name, value, acc) }
      case rw: TableIR => ab.result().foldRight[TableIR](rw) { case ((name, value), acc) => RelationalLetTable(name, value, acc) }
      case rw: MatrixIR => ab.result().foldRight[MatrixIR](rw) { case ((name, value), acc) => RelationalLetMatrixTable(name, value, acc) }
      case rw: BlockMatrixIR => ab.result().foldRight[BlockMatrixIR](rw) { case ((name, value), acc) => RelationalLetBlockMatrix(name, value, acc) }
    }

  }
}

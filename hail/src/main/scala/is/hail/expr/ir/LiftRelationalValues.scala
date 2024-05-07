package is.hail.expr.ir

import is.hail.types.virtual.TVoid
import is.hail.utils.BoxedArrayBuilder

import scala.annotation.tailrec
import scala.collection.mutable

object LiftRelationalValues {

  def apply(ir0: BaseIR): BaseIR = {

    def rewrite(ir: BaseIR, ab: BoxedArrayBuilder[(String, IR)], memo: mutable.Map[IR, String])
      : BaseIR = ir match {
      case RelationalLet(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        ab += name -> value2
        rewrite(body, ab, memo)
      case RelationalLetTable(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        ab += name -> value2
        rewrite(body, ab, memo)
      case RelationalLetMatrixTable(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        ab += name -> value2
        rewrite(body, ab, memo)
      case RelationalLetBlockMatrix(name, value, body) =>
        val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
        ab += name -> value2
        rewrite(body, ab, memo)
      case LiftMeOut(child) =>
        val name = memo.get(child) match {
          case Some(name) => name
          case None =>
            val name = genUID()
            val newChild = rewrite(child, ab, memo).asInstanceOf[IR]
            ab += name -> newChild
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
        val newChild = ir.mapChildren(rewrite(_, ab, memo))
        ab += ref.name -> newChild.asInstanceOf[IR]
        ref
      case x =>
        x.mapChildren(rewrite(_, ab, memo))
    }

    val ab = new BoxedArrayBuilder[(String, IR)]
    val memo = mutable.Map.empty[IR, String]

    @tailrec def unwrap(ir: BaseIR): BaseIR = ir match {
      case Block(Seq(), body) => unwrap(body)
      case x => x
    }

    val res = rewrite(unwrap(ir0), ab, memo) match {
      case rw: IR => ab.result().foldRight[IR](rw) { case ((name, value), acc) =>
          RelationalLet(name, value, acc)
        }
      case rw: TableIR => ab.result().foldRight[TableIR](rw) { case ((name, value), acc) =>
          RelationalLetTable(name, value, acc)
        }
      case rw: MatrixIR => ab.result().foldRight[MatrixIR](rw) { case ((name, value), acc) =>
          RelationalLetMatrixTable(name, value, acc)
        }
      case rw: BlockMatrixIR => ab.result().foldRight[BlockMatrixIR](rw) {
          case ((name, value), acc) => RelationalLetBlockMatrix(name, value, acc)
        }
    }
//    println(s"before:\n${Pretty.sexprStyle(unwrap(ir0))}")
//    println(s"after:\n${Pretty.sexprStyle(res)}")
    res

  }
}

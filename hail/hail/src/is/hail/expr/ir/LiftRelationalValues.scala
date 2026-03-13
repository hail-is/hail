package is.hail.expr.ir

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.compat.mutable.Growable
import is.hail.expr.ir.defs.{
  BlockMatrixCollect, BlockMatrixToValueApply, RelationalLet, RelationalRef,
  TableAggregate, TableCollect, TableCount, TableGetGlobals, TableToValueApply,
}
import is.hail.types.virtual.TVoid

import scala.collection.mutable

object LiftRelationalValues {

  def apply(ir0: BaseIR): BaseIR = {

    def rewrite(ir: BaseIR, ab: Growable[(Name, IR)], memo: mutable.Map[IR, Name]): BaseIR =
      ir match {
        case RelationalLet(name, value, body) =>
          val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
          val ab2 = ArraySeq.newBuilder[(Name, IR)]
          val memo2 = mutable.Map.empty[IR, Name]
          val body2 = rewrite(body, ab2, memo2).asInstanceOf[IR]
          RelationalLet(
            name,
            value2,
            ab2.result().foldRight[IR](body2) { case ((name, value), acc) =>
              RelationalLet(name, value, acc)
            },
          )
        case RelationalLetTable(name, value, body) =>
          val value2 = rewrite(value, ab, memo).asInstanceOf[IR]
          val ab2 = ArraySeq.newBuilder[(Name, IR)]
          val memo2 = mutable.Map.empty[IR, Name]
          val body2 = rewrite(body, ab2, memo2).asInstanceOf[TableIR]
          RelationalLetTable(
            name,
            value2,
            ab2.result().foldRight[TableIR](body2) { case ((name, value), acc) =>
              RelationalLetTable(name, value, acc)
            },
          )
        case (_: TableAggregate
            | _: TableCount
            | _: TableToValueApply
            | _: BlockMatrixToValueApply
            | _: TableCollect
            | _: BlockMatrixCollect
            | _: TableGetGlobals) if ir.typ != TVoid =>
          val name = memo.get(ir.asInstanceOf[IR]) match {
            case Some(name) => name
            case None =>
              val name = freshName()
              val newChild = ir.mapChildren(rewrite(_, ab, memo))
              ab += name -> newChild.asInstanceOf[IR]
              name
          }
          RelationalRef(name, ir.asInstanceOf[IR].typ)
        case x =>
          x.mapChildren(rewrite(_, ab, memo))
      }

    val ab = ArraySeq.newBuilder[(Name, IR)]
    val memo = mutable.Map.empty[IR, Name]
    rewrite(ir0, ab, memo) match {
      case rw: IR => ab.result().foldRight[IR](rw) { case ((name, value), acc) =>
          RelationalLet(name, value, acc)
        }
      case rw: TableIR => ab.result().foldRight[TableIR](rw) { case ((name, value), acc) =>
          RelationalLetTable(name, value, acc)
        }
      case rw: MatrixIR => ab.result().foldRight[MatrixIR](rw) { case ((name, value), acc) =>
          RelationalLetMatrixTable(name, value, acc)
        }
      case rw: BlockMatrixIR =>
        assert(ab.result().isEmpty)
        rw
    }

  }
}

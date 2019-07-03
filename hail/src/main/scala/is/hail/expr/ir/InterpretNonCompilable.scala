package is.hail.expr.ir

import is.hail.expr.types.virtual.TStruct
import org.apache.spark.sql.Row

object InterpretNonCompilable {

  def apply(ctx: ExecuteContext, ir: IR): (IR, Row, TStruct, String) = {

    val name = genUID()
    val nonCompilableNodes = LiftNonCompilable.extractNonCompilable(ir).toArray
    val emittable = nonCompilableNodes.filter { case (_, value) => CanEmit(value.typ) }
    val nonEmittable = nonCompilableNodes.filter { case (_, value) => !CanEmit(value.typ) }

    val rowType = TStruct(nonEmittable.map { case (k, v) => (k, v.typ)}: _*)
    val nonEmittableValues = Row.fromSeq(nonEmittable.map(_._2).map(Interpret[Any](ctx, _, optimize = false)))

    val nonEmittableMap = nonEmittable.map { case (k, v) => (v, GetField(Ref(name, rowType), k)) }.toMap
    val emittableMap = emittable.map { case (k, v) => (v, Literal.coerce(v.typ, Interpret(ctx, v, optimize = false))) }.toMap
    val jointMap = emittableMap ++ nonEmittableMap

    def rewrite(ir: IR): IR = {
      jointMap.get(ir) match {
        case Some(binding) => binding
        case None => MapIR(rewrite)(ir)
      }
    }

    (rewrite(ir), nonEmittableValues, rowType, name)
  }
}

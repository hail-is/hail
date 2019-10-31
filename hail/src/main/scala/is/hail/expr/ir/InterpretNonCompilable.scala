package is.hail.expr.ir

import is.hail.expr.types.virtual.TStruct
import org.apache.spark.sql.Row

import scala.collection.mutable

object InterpretNonCompilable {

  def extractNonCompilable(irs: IR*): Map[String, IR] = {
    val included = mutable.Set.empty[IR]

    def visit(ir: IR): Unit = {
      if (!Compilable(ir) && !included.contains(ir))
        included += ir
      ir.children.foreach {
        case ir: IR => visit(ir)
        case _ =>
      }
    }

    irs.foreach(visit)
    included.toArray.map { l => genUID() -> l }.toMap
  }

  def apply(ctx: ExecuteContext, ir: IR): (IR, Row, TStruct, String) = {

    val name = genUID()
    val nonCompilableNodes = extractNonCompilable(ir).toArray
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

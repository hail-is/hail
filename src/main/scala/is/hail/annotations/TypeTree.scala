package is.hail.annotations

import java.lang.reflect.Constructor

import is.hail.expr.{TContainer, TStruct, Type}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.{ClassTag, classTag}

object BroadcastTypeTree {
  def apply(sc: SparkContext, tt: TypeTree): Broadcast[TypeTree] = sc.broadcast(tt)

  def apply(sc: SparkContext, t: Type): Broadcast[TypeTree] = {
    t match {
      case TStruct(fields) =>
        BroadcastTypeTree(sc,
          new TypeTree(t,
            fields.map { f =>
              BroadcastTypeTree(sc, f.typ)
            }.toArray))

      case t: TContainer =>
        BroadcastTypeTree(sc,
          new TypeTree(t, Array(BroadcastTypeTree(sc, t.elementType))))

      case _ => null
    }
  }
}

class TypeTree(val typ: Type,
  subtrees: Array[Broadcast[TypeTree]]) {
  def subtree(i: Int): Broadcast[TypeTree] = subtrees(i)
}

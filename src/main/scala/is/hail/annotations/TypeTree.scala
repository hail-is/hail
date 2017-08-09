package is.hail.annotations

import java.lang.reflect.Constructor

import is.hail.expr.{TContainer, TStruct, Type}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.{ClassTag, classTag}

object BroadcastTypeTree {
  private val bcConstructor: Constructor[_] = {
    val torr = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    torr.getDeclaredConstructor(classOf[AnyRef], classOf[Long], classOf[ClassTag[_]])
  }

  private val m = new java.util.HashMap[Long, Broadcast[TypeTree]]

  def lookupBroadcast(id: Long): Broadcast[TypeTree] = {
    if (m.containsKey(id))
      m.get(id)
    else {
      val tbc = bcConstructor.newInstance(
        null: AnyRef,
        id: java.lang.Long,
        classTag[TypeTree]).asInstanceOf[Broadcast[TypeTree]]
      assert(tbc.value != null)
      m.put(id, tbc)
      tbc
    }
  }

  def apply(sc: SparkContext, tt: TypeTree): BroadcastTypeTree = {
    val bc = sc.broadcast(tt)
    new BroadcastTypeTree(bc, bc.id)
  }

  def apply(sc: SparkContext, t: Type): BroadcastTypeTree = {
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

class BroadcastTypeTree(@transient var bc: Broadcast[TypeTree],
  id: Long) extends Serializable {
  def value: TypeTree = {
    if (bc == null)
      bc = BroadcastTypeTree.lookupBroadcast(id)
    bc.value
  }
}

class TypeTree(val typ: Type,
  subtrees: Array[BroadcastTypeTree]) {
  def subtree(i: Int): BroadcastTypeTree = subtrees(i)
}

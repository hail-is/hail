package is.hail.methods

import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

import scala.collection.mutable
import scala.reflect.ClassTag

import is.hail.utils.dictionaryOrdering

object MaximalIndependentSet {

  def apply[VD: ClassTag, ED: ClassTag](g: Graph[VD, ED], optComparisonFunction: Option[(Int, Int) => Int] = None): Set[Long] = {

    type Message = (Int, VertexId)

    val cmp: (Int, Int) => Int= optComparisonFunction.getOrElse((_,_) => 0)
    val cmpOrd = new Ordering[Int] { def compare(x: Int, y: Int): Int = cmp(x, y) }
    val ord = dictionaryOrdering[Message](
      Ordering.by(_._1),
      Ordering.by[Message, Int](_._2.toInt)(cmpOrd),
      Ordering.by(_._2))

    import ord.mkOrderingOps

    val initialMsg = (-1, -1L)

    def receiveMessage(vertexId: VertexId, value: Message, message: Message): Message = {
      value max message
    }

    def sendMsg(triplet: EdgeTriplet[Message, ED]): Iterator[(VertexId, Message)] = {
      if (triplet.srcAttr > triplet.dstAttr)
        Iterator((triplet.dstId, triplet.srcAttr))
      else if (triplet.srcAttr < triplet.dstAttr)
        Iterator((triplet.srcId, triplet.dstAttr))
      else
        Iterator.empty
    }

    def mergeMsg(x: Message, y: Message) =
      x max y

    def updateVertexDegrees(toBeComputed: Graph[_, ED]): Graph[Message, ED] = {
      Graph(toBeComputed.vertices.leftZipJoin(toBeComputed.degrees) { (v, _, degree) => (degree.getOrElse(0), v) }, toBeComputed.edges)
    }

    val edgeDirection = EdgeDirection.Either

    var idSetToDelete = mutable.Set[VertexId]()

    g.persist()
    var workingGraph = g

    while (workingGraph.numEdges > 0) {
      idSetToDelete ++= updateVertexDegrees(workingGraph)
          .pregel(initialMsg, Int.MaxValue, edgeDirection)(receiveMessage, sendMsg, mergeMsg)
          .vertices
          .filter{ case (id, (maxDegrees, maxID)) => maxID == id && maxDegrees != 0}
          .map(_._1)
          .collect()
          .toSet
      workingGraph.unpersist()
      workingGraph = g.subgraph(_ => true, (id, value) => !idSetToDelete.contains(id))
      workingGraph.persist()
    }

    val survivors = workingGraph.vertices.keys.collect().toSet
    g.unpersist()
    workingGraph.unpersist()

    survivors
  }


  def ofIBDMatrix(inputRDD: RDD[((Int, Int), Double)], thresh: Double, vertexIDs: Seq[Int], comparisonFunction: Option[(Int, Int) => Int] = None): Set[Long] = {
    val sc = inputRDD.sparkContext

    val filteredRDD = inputRDD.filter(_._2 >= thresh)

    val edges: RDD[Edge[Double]] = filteredRDD.map{ case ((v1, v2), weight) => Edge(v1, v2, weight)}

    val vertices: RDD[(VertexId, Null)] = sc.parallelize(vertexIDs).map(id => (id, null))

    val graph: Graph[Null, Double] = Graph(vertices, edges)

    apply(graph, comparisonFunction)
  }
}

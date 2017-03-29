package is.hail.methods

import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

import scala.collection.mutable
import scala.reflect.ClassTag

object MaximalIndependentSet {

  def apply[VD: ClassTag, ED: ClassTag](g: Graph[VD, ED], optComparisonFunction: Option[(Int, Int) => Int] = None): Set[Long] = {

    val comparisonFunction: (Int, Int) => Int = optComparisonFunction.getOrElse((_, _) => 0)
    type Message = (Int, VertexId)

    //val subOrd = Ordering.by[Message, Int]
    //val ord: Ordering[Message] = Ordering.by[Message, Int](_._1).thenComparing(Ordering.by[Message, Int](_._2.toInt))
    val ord = new Ordering[Message] {
      override def compare(x: (Int, VertexId), y: (Int, VertexId)): Int = {
        //By degree
        (x._1 compare y._1) match {
          case 0 => {
            //By function
            comparisonFunction(x._2.toInt, y._2.toInt) match {
              //By VertexID
              case 0 => x._2 compare y._2
              case c => c
            }
          }
          case c => c
        }
      }
    }
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

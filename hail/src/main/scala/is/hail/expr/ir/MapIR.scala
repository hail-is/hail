package is.hail.expr.ir

import is.hail.utils.{IntArrayStack, ObjectArrayStack}

object MapIR {
  def apply(f: IR => IR)(ir: IR): IR = ir match {
    case ta: TableAggregate => ta
    case ma: MatrixAggregate => ma
    case _ => Copy(ir, Children(ir).map {
      case c: IR => f(c)
      case c => c
    })
  }

  def mapBaseIR(ir: BaseIR, f: BaseIR => BaseIR): BaseIR = f(ir.copy(newChildren = ir.children.map(mapBaseIR(_, f))))
}

object VisitIR {
  def topDown(ir: BaseIR)(f: BaseIR => Unit): Unit = {
    val workQueue = new java.util.ArrayDeque[BaseIR]()

    workQueue.addLast(ir)

    while (!workQueue.isEmpty) {

      val toProcess = workQueue.removeFirst()
      f(toProcess)
      toProcess.children.foreach(workQueue.addLast)
    }
  }

  def bottomUp(ir: BaseIR)(f: BaseIR => Unit): Unit = {
    val workStack = new ObjectArrayStack[BaseIR]()
    val idxStack = new IntArrayStack()

    workStack.push(ir)
    idxStack.push(0)

    while (!workStack.isEmpty) {
      val currNode = workStack.pop()
      val currIdx = idxStack.pop()

      val children = currNode.children
      if (currIdx == children.length)
        f(currNode)
      else {
        workStack.push(currNode)
        idxStack.push(currIdx + 1)
        workStack.push(children(currIdx))
        idxStack.push(0)
      }
    }
  }
}
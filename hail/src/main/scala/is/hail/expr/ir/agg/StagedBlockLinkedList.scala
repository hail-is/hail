package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitTriplet, EmitFunctionBuilder}
import is.hail.expr.types.physical._
import is.hail.utils._

object StagedBlockLinkedList {
  val defaultBlockCap: Int = 64
}

class StagedBlockLinkedList(val elemType: PType, val fb: EmitFunctionBuilder[_]) {
  import StagedBlockLinkedList._

  val firstNode = fb.newField[Long]
  val lastNode = fb.newField[Long]
  val totalCount = fb.newField[Int]

  val storageType = PStruct(
    "firstNode" -> PInt64Required,
    "lastNode" -> PInt64Required,
    "totalCount" -> PInt32Required)

  def load(src: Code[Long]): Code[Unit] = Code(
    firstNode := Region.loadAddress(storageType.fieldOffset(src, 0)),
    lastNode := Region.loadAddress(storageType.fieldOffset(src, 1)),
    totalCount := Region.loadInt(storageType.fieldOffset(src, 2)))

  def store(dst: Code[Long]): Code[Unit] = Code(
    Region.storeAddress(storageType.fieldOffset(dst, 0), firstNode),
    Region.storeAddress(storageType.fieldOffset(dst, 1), lastNode),
    Region.storeInt(storageType.fieldOffset(dst, 2), totalCount))

  val i = fb.newField[Int]
  val p = fb.newField[Boolean]
  val tmpNode = fb.newField[Long]

  type Node = Code[Long]

  val bufferType = PArray(elemType, required = true)

  val nodeType = PStruct(
    "buf" -> bufferType,
    "count" -> PInt32Required,
    "next" -> PInt64Optional)

  private def buffer(n: Node): Code[Long] =
    Region.loadAddress(nodeType.fieldOffset(n, 0))

  private[agg] def capacity(n: Node): Code[Int] =
    bufferType.loadLength(buffer(n))

  private def count(n: Node): Code[Int] =
    Region.loadInt(nodeType.fieldOffset(n, 1))

  private def incrCount(n: Node): Code[Unit] =
    Region.storeInt(nodeType.fieldOffset(n, 1), count(n) + 1)

  private def next(n: Node): Node =
    Region.loadAddress(nodeType.fieldOffset(n, 2))

  private def hasNext(n: Node): Code[Boolean] =
    nodeType.isFieldDefined(n, 2)

  private def setNext(n: Node, nNext: Node): Code[Unit] = Code(
    nodeType.setFieldPresent(n, 2),
    Region.storeAddress(nodeType.fieldOffset(n, 2), nNext))

  private def initNode(n: Node, buf: Code[Long], count: Code[Int]): Code[Unit] =
    Code(
      Region.storeAddress(nodeType.fieldOffset(n, 0), buf),
      Region.storeInt(nodeType.fieldOffset(n, 1), count),
      nodeType.setFieldMissing(n, 2))

  private def pushPresent(n: Node, store: Code[Long] => Code[Unit]): Code[Unit] =
    Code(
      if(elemType.required) Code._empty else bufferType.setElementPresent(buffer(n), count(n)),
      store(bufferType.elementOffset(buffer(n), capacity(n), count(n))),
      incrCount(n))

  private def pushMissing(n: Node): Code[Unit] = {
    if(elemType.required)
      Code._fatal("cannot push missing onto list with required elements")
    else
      Code(
        bufferType.setElementMissing(buffer(n), count(n)),
        incrCount(n))
  }

  private def allocateNode(r: Code[Region], cap: Code[Int]): (Code[Unit], Node) = {
    val setup = Code(
      tmpNode := r.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(tmpNode,
        buf = r.allocate(bufferType.contentsAlignment, bufferType.contentsByteSize(cap)),
        count = 0),
      bufferType.stagedInitialize(buffer(tmpNode), cap))
    (setup, tmpNode)
  }

  def initWithCapacity(r: Code[Region], initialCap: Code[Int]): Code[Unit] = {
    val (setup, n) = allocateNode(r, initialCap)
    Code(setup,
      firstNode := n,
      lastNode := n,
      totalCount := 0)
  }

  def init(r: Code[Region]): Code[Unit] =
    initWithCapacity(r, defaultBlockCap)

  private[agg] def pushNewBlockNode(r: Code[Region], cap: Code[Int]): Code[Unit] = {
    val (setup, newNode) = allocateNode(r, cap)
    Code(setup,
      setNext(lastNode, newNode),
      lastNode := newNode)
  }

  private[agg] def foreachNode(f: Node => Code[Unit]): Code[Unit] = {
    Code(
      tmpNode := firstNode,
      p := true,
      Code.whileLoop(p,
        f(tmpNode),
        p := hasNext(tmpNode),
        tmpNode := next(tmpNode)))
  }

  def foreach(f: EmitTriplet => Code[Unit]): Code[Unit] = {
    def et(n: Node, i: Code[Int]) =
      EmitTriplet(Code._empty,
        bufferType.isElementMissing(buffer(n), i),
        Region.loadIRIntermediate(elemType)(
          bufferType.elementOffset(buffer(n), capacity(n), i)))
    foreachNode { n => Code(
      i := 0,
      Code.whileLoop(i < count(n),
        f(et(n, i)),
        i := i + 1))
    }
  }

  def push(r: Code[Region], elt: EmitTriplet): Code[Unit] = {
    Code(
      (count(lastNode) >= capacity(lastNode)).orEmpty(
        pushNewBlockNode(r, defaultBlockCap)), // push a new block if lastNode is full
      elt.setup,
      elt.m.mux(
        pushMissing(lastNode),
        pushPresent(lastNode, { dst =>
          if(elemType.isPrimitive)
            Region.storePrimitive(elemType, dst)(elt.value)
          else
            StagedRegionValueBuilder.deepCopy(fb, r, elemType, elt.value, dst)
        })),
      totalCount := totalCount + 1)
  }

  def append(r: Code[Region], bll: StagedBlockLinkedList): Code[Unit] = {
    assert(bll.elemType.isOfType(elemType))
    bll.foreach(push(r, _))
  }

  def appendShallow(r: Code[Region], atyp: PArray, aoff: Code[Long]): Code[Unit] = {
    assert(atyp.isOfType(bufferType))
    assert(atyp.elementType.required == elemType.required)
    val len = atyp.loadLength(r, aoff)
    Code(
      tmpNode := r.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(tmpNode,
        buf = aoff,
        count = len),
      setNext(lastNode, tmpNode),
      lastNode := tmpNode,
      totalCount := totalCount + len)
  }

  def writeToSRVB(srvb: StagedRegionValueBuilder): Code[Unit] = {
    assert(srvb.typ.isOfType(bufferType))
    Code(
      srvb.start(totalCount, init = true),
      foreach { elt =>
        Code(
          elt.m.mux(
            srvb.setMissing(),
            srvb.addWithDeepCopy(elemType, elt.value)),
          srvb.advance())
      })
  }

  def toArray: Code[Long] = {
    val srvb = new StagedRegionValueBuilder(fb, bufferType)
    Code(writeToSRVB(srvb), srvb.end())
  }
}

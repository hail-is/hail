package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitTriplet, EmitRegion, EmitMethodBuilder}
import is.hail.expr.types.physical._
import is.hail.utils._

object StagedBlockLinkedList {
  val defaultBlockCap: Int = 64

  // src : Code[PointerTo[T]]
  // dst : Code[PointerTo[T]]
  private def deepCopy(er: EmitRegion, typ: PType, src: Code[Long], dst: Code[Long]): Code[Unit] =
    storeDeepCopy(er, typ, Region.loadIRIntermediate(typ)(src), dst)

  // v : Code[T]
  // dst : Code[PointerTo[T]]
  private def storeDeepCopy(er: EmitRegion, typ: PType, v: Code[_], dst: Code[Long]): Code[Unit] = {
    val ftyp = typ.fundamentalType
    if (ftyp.isPrimitive)
      Region.storePrimitive(ftyp, dst)(v)
    else
      StagedRegionValueBuilder.deepCopy(er, typ, coerce[Long](v), dst)
  }
}

class StagedBlockLinkedList(val elemType: PType, val er: EmitRegion) {
  import StagedBlockLinkedList._
  private val EmitRegion(mb, region) = er

  val firstNode = mb.newField[Long]
  val lastNode = mb.newField[Long]
  val totalCount = mb.newField[Int]

  val i = mb.newField[Int]
  val t = mb.newField[Boolean]
  val tmpNode = mb.newField[Long]

  type Node = Code[Long]

  val bufferType = PArray(elemType.setRequired(false), required = true)

  val nodeType = PStruct(
    "buf" -> bufferType,
    "count" -> PInt32Required,
    "nextB" -> PInt64Optional)

  private def buffer(n: Node): Code[Long] = region.loadAddress(nodeType.fieldOffset(n, 0))
  private[agg] def capacity(n: Node): Code[Int] = bufferType.loadLength(region, buffer(n))

  private def count(n: Node): Code[Int] = region.loadInt(nodeType.fieldOffset(n, 1))
  private def incrCount(n: Node): Code[Unit] =
    region.storeInt(nodeType.fieldOffset(n, 1), count(n) + 1)

  private def next(n: Node): Node = region.loadAddress(nodeType.fieldOffset(n, 2))
  private def hasNext(n: Node): Code[Boolean] = nodeType.isFieldDefined(region, n, 2)
  private def setNext(n: Node, nNext: Node): Code[Unit] = Code(
    nodeType.setFieldPresent(region, n, 2),
    region.storeAddress(nodeType.fieldOffset(n, 2), nNext))

  private def initNode(n: Node, cap: Code[Int]): Code[Unit] = {
    val a = mb.newField[Int]
    Code(
      region.storeAddress(nodeType.fieldOffset(n, 0),
        region.allocate(bufferType.contentsAlignment, bufferType.contentsByteSize(cap))),
      bufferType.stagedInitialize(buffer(n), cap, setMissing = false),
      region.storeInt(nodeType.fieldOffset(n, 1), 0),
      nodeType.setFieldMissing(region, n, 2))
  }

  private def elemAddress(n: Node, i: Code[Int]): EmitTriplet =
    EmitTriplet(Code._empty,
      bufferType.isElementMissing(region, buffer(n), i),
      bufferType.elementOffsetInRegion(region, buffer(n), i))

  private def push(n: Node, store: Code[Long] => Code[Unit]): Code[Unit] = Code(
    // bufferType.setElementPresent(region, buffer(n), count(n)),
    // ^ don't need to set present because 'setMissing=false' in initNode
    store(bufferType.elementOffsetInRegion(region, buffer(n), count(n))),
    incrCount(n))

  private def pushMissing(n: Node): Code[Unit] = Code(
    bufferType.setElementMissing(region, buffer(n), count(n)),
    incrCount(n))

  private def allocateNode(cap: Code[Int]): (Code[Unit], Node) = {
    val setup = Code(
      tmpNode := region.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(tmpNode, cap))
    (setup, tmpNode)
  }

  def initWithCapacity(initialCap: Code[Int]): Code[Unit] = {
    val (setup, n) = allocateNode(initialCap)
    Code(setup,
      firstNode := n,
      lastNode := n,
      totalCount := 0)
  }

  lazy val init: Code[Unit] =
    initWithCapacity(defaultBlockCap)

  private[agg] def pushNewBlockNode(cap: Code[Int]): Code[Unit] = {
    val (setup, newNode) = allocateNode(cap)
    Code(setup,
      setNext(lastNode, newNode),
      lastNode := newNode)
  }

  private[agg] def foreachNode(f: Node => Code[Unit]): Code[Unit] = {
    Code(
      tmpNode := firstNode,
      t := true,
      Code.whileLoop(t,
        f(tmpNode),
        t := hasNext(tmpNode),
        tmpNode := next(tmpNode)))
  }

  def foreachElemAddress(f: EmitTriplet => Code[Unit]): Code[Unit] = {
    foreachNode { n => Code(
      i := 0,
      Code.whileLoop(i < count(n),
        f(elemAddress(n, i)),
        i := i + 1))
    }
  }

  def push(elt: EmitTriplet): Code[Unit] = {
    Code(
      (count(lastNode) >= capacity(lastNode)).mux(
        pushNewBlockNode(defaultBlockCap), // push a new block if lastNode is full
        Code._empty),
      elt.setup,
      elt.m.mux(
        pushMissing(lastNode),
        push(lastNode, storeDeepCopy(er, elemType, elt.v, _))),
      totalCount := totalCount + 1)
  }

  def pushFromAddress(addr: EmitTriplet): Code[Unit] =
    push(EmitTriplet(addr.setup, addr.m,
      Region.loadIRIntermediate(elemType)(addr.value)))

  def append(bll: StagedBlockLinkedList): Code[Unit] = {
    assert(bll.elemType.isOfType(elemType))
    bll.foreachElemAddress(pushFromAddress)
  }

  def writeToSRVB(srvb: StagedRegionValueBuilder): Code[Unit] = {
    assert(srvb.typ.isOfType(bufferType))
    Code(
      srvb.start(totalCount, init = true),
      foreachElemAddress { addr =>
        Code(
          addr.m.mux(
            srvb.setMissing(),
            deepCopy(er, elemType, addr.value, srvb.currentOffset)),
          srvb.advance())
      })
  }

  def toArray: Code[Long] = {
    val srvb = new StagedRegionValueBuilder(er, bufferType)
    Code(writeToSRVB(srvb), srvb.end())
  }
}

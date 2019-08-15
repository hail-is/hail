package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitTriplet, EmitRegion, EmitFunctionBuilder}
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.io.{OutputBuffer, InputBuffer, CodecSpec, EmitPackEncoder, EmitPackDecoder}

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

  private def allocateNode(dstNode: Settable[Long])(r: Code[Region], cap: Code[Int]): Code[Unit] =
    Code(
      dstNode := r.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(dstNode,
        buf = r.allocate(bufferType.contentsAlignment, bufferType.contentsByteSize(cap)),
        count = 0),
      bufferType.stagedInitialize(buffer(dstNode), cap))

  private def initWithCapacity(r: Code[Region], initialCap: Code[Int]): Code[Unit] = {
    Code(
      allocateNode(firstNode)(r, initialCap),
      lastNode := firstNode,
      totalCount := 0)
  }

  def init(r: Code[Region]): Code[Unit] =
    initWithCapacity(r, defaultBlockCap)

  private def pushNewBlockNode(r: Code[Region], cap: Code[Int]): Code[Unit] = {
    val newNode = fb.newField[Long]
    Code(
      allocateNode(newNode)(r, cap),
      setNext(lastNode, newNode),
      lastNode := newNode)
  }

  private def foreachNode(tmpNode: Settable[Long])(body: Code[Unit]): Code[Unit] = {
    val present = fb.newField[Boolean]
    Code(
      tmpNode := firstNode,
      present := true,
      Code.whileLoop(present,
        body,
        present := hasNext(tmpNode),
        tmpNode := next(tmpNode)))
  }

  def foreach(f: EmitTriplet => Code[Unit]): Code[Unit] = {
    val n = fb.newField[Long]
    val i = fb.newField[Int]
    val et =
      EmitTriplet(Code._empty,
        bufferType.isElementMissing(buffer(n), i),
        Region.loadIRIntermediate(elemType)(
          bufferType.elementOffset(buffer(n), capacity(n), i)))
    foreachNode(n) { Code(
      i := 0,
      Code.whileLoop(i < count(n),
        f(et),
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

  def serialize(r: Code[Region], ob: Code[OutputBuffer]): Code[Unit] = {
    val n = fb.newField[Long]
    val i = fb.newField[Int]
    Code(
      foreachNode(n) {
        // NOTE: we are only copying a portion of the missingness bytes, only as many as there are
        // present (specifically: 'count(n)', not 'capacity(n)'). we can't use emitArray because it
        // would mistakenly copy elements according to the capacity ...
        val nMissingBytes = (count(n) + 7) >> 3
        val writeMissingBytes = ob.writeBytes(r, buffer(n) + 4, nMissingBytes)
        val eltm = bufferType.isElementMissing(r, buffer(n), i)
        val eltOff = bufferType.elementOffsetInRegion(r, buffer(n), i)
        val writeElt = EmitPackEncoder.emit(elemType, elemType, fb.apply_method, r, eltOff, ob)
        Code(
          ob.writeBoolean(true),
          ob.writeInt(count(n)),
          if(elemType.required) Code._empty else writeMissingBytes,
          i := 0,
          Code.whileLoop(i < count(n), Code(
            if(elemType.required) writeElt else eltm.mux(Code._empty, writeElt),
            i := i + 1)))
      },
      ob.writeBoolean(false))
  }

  def deserialize(r: Code[Region], ib: Code[InputBuffer]): Code[Unit] = {
    // ... however, we can use EmitPackDecoder to decode an entire array and add it as a node
    val srvb = new StagedRegionValueBuilder(EmitRegion(fb.apply_method, r), bufferType)
    val bufFType = bufferType.fundamentalType
    Code.whileLoop(ib.readBoolean(),
      EmitPackDecoder.emitArray(bufFType, bufFType, fb.apply_method, ib, srvb),
      appendShallow(r, srvb.end()))
  }

  private def appendShallow(r: Code[Region], aoff: Code[Long]): Code[Unit] = {
    val len = bufferType.loadLength(r, aoff)
    val newNode = fb.newField[Long]
    Code(
      newNode := r.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(newNode,
        buf = aoff,
        count = len),
      setNext(lastNode, newNode),
      lastNode := newNode,
      totalCount := totalCount + len)
  }

  def initWithDeepCopy(r: Code[Region], other: StagedBlockLinkedList): Code[Unit] = {
    val i = fb.newField[Int]
    val buf = buffer(firstNode)
    val bufi = bufferType.elementOffsetInRegion(r, buf, i)
    Code(
      initWithCapacity(r, other.totalCount),
      i := 0,
      other.foreach { et =>
        Code(
          et.m.mux(bufferType.setElementMissing(r, buf, i),
            Code(
              bufferType.setElementPresent(r, buf, i),
              if(elemType.isPrimitive)
                Region.storePrimitive(elemType, bufi)(et.value)
              else
                StagedRegionValueBuilder.deepCopy(fb, r, elemType, et.value, bufi))),
          incrCount(firstNode),
          i := i + 1)
      },
      totalCount := other.totalCount)
  }
}

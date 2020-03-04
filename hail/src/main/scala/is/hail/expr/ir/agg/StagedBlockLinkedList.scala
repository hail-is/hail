package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.encoded._
import is.hail.expr.types.physical._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

object StagedBlockLinkedList {
  val defaultBlockCap: Int = 64

  private val nil: Long = 0L
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
    firstNode := Region.loadAddress(storageType.fieldOffset(src, "firstNode")),
    lastNode := Region.loadAddress(storageType.fieldOffset(src, "lastNode")),
    totalCount := Region.loadInt(storageType.fieldOffset(src, "totalCount")))

  def store(dst: Code[Long]): Code[Unit] = Code(
    Region.storeAddress(storageType.fieldOffset(dst, "firstNode"), firstNode),
    Region.storeAddress(storageType.fieldOffset(dst, "lastNode"), lastNode),
    Region.storeInt(storageType.fieldOffset(dst, "totalCount"), totalCount))

  type Node = Code[Long]

  val bufferType = PArray(elemType, required = true)
  val bufferEType = EType.defaultFromPType(bufferType).asInstanceOf[EArray]

  val nodeType = PStruct(
    "buf" -> bufferType,
    "count" -> PInt32Required,
    "next" -> PInt64Required)

  private def buffer(n: Node): Code[Long] =
    Region.loadAddress(nodeType.fieldOffset(n, "buf"))

  private[agg] def capacity(n: Node): Code[Int] =
    bufferType.loadLength(buffer(n))

  private def count(n: Node): Code[Int] =
    Region.loadInt(nodeType.fieldOffset(n, "count"))

  private def incrCount(n: Node): Code[Unit] =
    Region.storeInt(nodeType.fieldOffset(n, "count"), count(n) + 1)

  private def next(n: Node): Node =
    Region.loadAddress(nodeType.fieldOffset(n, "next"))

  private def hasNext(n: Node): Code[Boolean] =
    next(n) cne nil

  private def setNext(n: Node, nNext: Node): Code[Unit] =
    Region.storeAddress(nodeType.fieldOffset(n, "next"), nNext)

  private def initNode(n: Node, buf: Code[Long], count: Code[Int]): Code[Unit] =
    Code(
      Region.storeAddress(nodeType.fieldOffset(n, "buf"), buf),
      Region.storeInt(nodeType.fieldOffset(n, "count"), count),
      Region.storeAddress(nodeType.fieldOffset(n, "next"), nil))

  private def pushPresent(n: Node, store: Code[Long] => Code[Unit]): Code[Unit] =
    Code(
      if(elemType.required) Code._empty else bufferType.setElementPresent(buffer(n), count(n)),
      store(bufferType.elementOffset(buffer(n), capacity(n), count(n))),
      incrCount(n))

  private def pushMissing(n: Node): Code[Unit] =
    Code(
      bufferType.setElementMissing(buffer(n), count(n)),
      incrCount(n))

  private def allocateNode(dstNode: Settable[Long])(r: Code[Region], cap: Code[Int]): Code[Unit] =
    Code(
      dstNode := r.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(dstNode,
        buf = bufferType.allocate(r, cap),
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

  private def pushNewBlockNode(mb: MethodBuilder, r: Code[Region], cap: Code[Int]): Code[Unit] = {
    val newNode = mb.newLocal[Long]
    Code(
      allocateNode(newNode)(r, cap),
      setNext(lastNode, newNode),
      lastNode := newNode)
  }

  private def foreachNode(mb: MethodBuilder, tmpNode: Settable[Long])(body: Code[Unit]): Code[Unit] = {
    val present = mb.newLocal[Boolean]
    Code(
      tmpNode := firstNode,
      present := true,
      Code.whileLoop(present,
        body,
        present := hasNext(tmpNode),
        tmpNode := next(tmpNode)))
  }

  private def foreach(mb: MethodBuilder)(f: EmitTriplet => Code[Unit]): Code[Unit] = {
    val n = mb.newLocal[Long]
    foreachNode(mb, n) {
      val i = mb.newLocal[Int]
      val bufim = bufferType.isElementMissing(buffer(n), i)
      val bufiv = Region.loadIRIntermediate(elemType)(bufferType.elementOffset(buffer(n), capacity(n), i))
      Code(
        i := 0,
        Code.whileLoop(i < count(n),
          f(EmitTriplet(Code._empty, bufim, PValue(elemType, bufiv))),
          i := i + 1))
    }
  }

  private def push(mb: MethodBuilder, r: Code[Region], m: Code[Boolean], v: Code[_]): Code[Unit] = {
    var push = pushPresent(lastNode, StagedRegionValueBuilder.deepCopy(fb, r, elemType, v, _))
    if(!elemType.required)
      push = m.mux(pushMissing(lastNode), push)
    Code(
      (count(lastNode) >= capacity(lastNode)).orEmpty(
        pushNewBlockNode(mb, r, defaultBlockCap)), // push a new block if lastNode is full
      push,
      totalCount := totalCount + 1)
  }

  def push(region: Code[Region], elt: EmitTriplet): Code[Unit] = {
    val eltTI = typeToTypeInfo(elemType)
    val pushF = fb.newMethod("blockLinkedListPush",
      Array[TypeInfo[_]](typeInfo[Region], typeInfo[Boolean], eltTI),
      typeInfo[Unit])
    pushF.emit(push(pushF,
      pushF.getArg[Region](1),
      pushF.getArg[Boolean](2),
      pushF.getArg(3)(eltTI)))
    Code(
      elt.setup,
      elt.m.mux(
        pushF.invoke(region, true, defaultValue(elemType)),
        pushF.invoke(region, false, elt.v)))
  }

  def append(region: Code[Region], bll: StagedBlockLinkedList): Code[Unit] = {
    // it would take additional logic to get self-append to work, but we don't need it to anyways
    assert(bll ne this)
    assert(bll.elemType.isOfType(elemType))
    val appF = fb.newMethod("blockLinkedListAppend",
      Array[TypeInfo[_]](typeInfo[Region]),
      typeInfo[Unit])
    appF.emit(bll.foreach(appF) { elt =>
      push(appF, appF.getArg[Region](1), elt.m, elt.v)
    })
    appF.invoke(region)
  }

  def writeToSRVB(srvb: StagedRegionValueBuilder): Code[Unit] = {
    assert(srvb.typ.fundamentalType.isOfType(bufferType.fundamentalType), s"srvb: ${srvb.typ}, buf: ${bufferType.fundamentalType}")
    val writeF = fb.newMethod("blockLinkedListToSRVB", Array[TypeInfo[_]](), typeInfo[Unit])
    writeF.emit {
      Code(
        srvb.start(totalCount, init = true),
        foreach(writeF) { elt =>
          Code(
            elt.m.mux(
              srvb.setMissing(),
              srvb.addWithDeepCopy(elemType, elt.value)),
            srvb.advance())
        })
    }
    writeF.invoke()
  }

  def serialize(region: Code[Region], outputBuffer: Code[OutputBuffer]): Code[Unit] = {
    val serF = fb.newMethod("blockLinkedListSerialize",
      Array[TypeInfo[_]](typeInfo[Region], typeInfo[OutputBuffer]),
      typeInfo[Unit])
    val ob = serF.getArg[OutputBuffer](2)
    serF.emit {
      val n = serF.newLocal[Long]
      val i = serF.newLocal[Int]
      val b = serF.newLocal[Long]
      Code(
        foreachNode(serF, n) { Code(
          ob.writeBoolean(true),
          b := buffer(n),
          bufferEType.buildPrefixEncoder(bufferType.fundamentalType, serF, b, ob, count(n)))
        },
        ob.writeBoolean(false))
    }
    serF.invoke(region, outputBuffer)
  }

  def deserialize(region: Code[Region], inputBuffer: Code[InputBuffer]): Code[Unit] = {
    val desF = fb.newMethod("blockLinkedListDeserialize",
      Array[TypeInfo[_]](typeInfo[Region], typeInfo[InputBuffer]),
      typeInfo[Unit])
    val r = desF.getArg[Region](1).load
    val ib = desF.getArg[InputBuffer](2).load
    val array = desF.newLocal[Long]("array")
    val bufFType = bufferType.fundamentalType
    val dec = bufferEType.buildDecoder(bufferType, desF)
    desF.emit(
      Code.whileLoop(ib.readBoolean(),
        array := dec(r, ib),
        appendShallow(desF, r, array))
    )
    desF.invoke(region, inputBuffer)
  }

  private def appendShallow(mb: MethodBuilder, r: Code[Region], aoff: Code[Long]): Code[Unit] = {
    val len = bufferType.loadLength(aoff)
    val newNode = mb.newLocal[Long]
    Code(
      newNode := r.allocate(nodeType.alignment, nodeType.byteSize),
      initNode(newNode,
        buf = aoff,
        count = len),
      setNext(lastNode, newNode),
      lastNode := newNode,
      totalCount := totalCount + len)
  }

  def initWithDeepCopy(region: Code[Region], other: StagedBlockLinkedList): Code[Unit] = {
    assert(other ne this)
    assert(other.fb eq fb)
    val initF = fb.newMethod("blockLinkedListDeepCopy",
      Array[TypeInfo[_]](typeInfo[Region]),
      typeInfo[Unit])
    val r = initF.getArg[Region](1)
    initF.emit {
      val i = initF.newLocal[Int]
      val buf = buffer(firstNode)
      val bufi = bufferType.elementOffset(buf, i)
      Code(
        initWithCapacity(r, other.totalCount),
        i := 0,
        other.foreach(initF) { et =>
          Code(
            et.m.mux(bufferType.setElementMissing(buf, i),
              Code(
                bufferType.setElementPresent(buf, i),
                StagedRegionValueBuilder.deepCopy(fb, r, elemType, et.value, bufi))),
            incrCount(firstNode),
            i := i + 1)
        },
        totalCount := other.totalCount)
    }
    initF.invoke(region)
  }
}

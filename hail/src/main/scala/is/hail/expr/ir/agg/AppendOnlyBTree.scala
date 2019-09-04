package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.asm4s.coerce
import is.hail.io.{InputBuffer, OutputBuffer}

trait BTreeKey {
  def storageType: PType
  def compType: PType
  def isEmpty(off: Code[Long]): Code[Boolean]
  def initializeEmpty(off: Code[Long]): Code[Unit]

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit]
  def deepCopy(er: EmitRegion, src: Code[Long], dest: Code[Long]): Code[Unit]

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int]
  def loadCompKey(off: Code[Long]): (Code[Boolean], Code[_])

  def compSame(off: Code[Long], other: Code[Long]): Code[Int] =
    compKeys(loadCompKey(off), loadCompKey(other))
  def compWithKey(off: Code[Long], k: (Code[Boolean], Code[_])): Code[Int] =
    compKeys(loadCompKey(off), k)
}

class AppendOnlyBTree(fb: EmitFunctionBuilder[_], key: BTreeKey, region: Code[Region], root: ClassFieldRef[Long], maxElements: Int = 2) {
  private val splitIdx = maxElements / 2
  private val eltType: PTuple = PTuple(key.storageType, PInt64(true))
  private val elementsType: PTuple = PTuple(required = true, Array.fill(maxElements)(eltType): _*)
  private val storageType: PStruct = PStruct(required = true,
    "parent" -> PInt64(),
    "child0" -> PInt64(),
    "elements" -> elementsType)

  private def createNode(nodeBucket: Settable[Long]): Code[Unit] = Code(
    nodeBucket := region.allocate(storageType.alignment, storageType.byteSize),
    storageType.setAllMissing(nodeBucket),
    elementsType.setAllMissing(elements(nodeBucket)))

  private def isRoot(node: Code[Long]): Code[Boolean] = storageType.isFieldMissing(node, 0)
  private def isLeaf(node: Code[Long]): Code[Boolean] = storageType.isFieldMissing(node, 1)
  private def getParent(node: Code[Long]): Code[Long] = Region.loadAddress(storageType.loadField(node, 0))

  private def elements(node: Code[Long]): Code[Long] = storageType.loadField(node, 2)
  private def hasKey(node: Code[Long], i: Int): Code[Boolean] = elementsType.isFieldDefined(elements(node), i)
  private def setKeyPresent(node: Code[Long], i: Int): Code[Unit] = elementsType.setFieldPresent(elements(node), i)
  private def setKeyMissing(node: Code[Long], i: Int): Code[Unit] = elementsType.setFieldMissing(elements(node), i)
  private def isFull(node: Code[Long]): Code[Boolean] = hasKey(node, maxElements - 1)
  private def keyOffset(node: Code[Long], i: Int): Code[Long] = eltType.fieldOffset(elementsType.loadField(elements(node), i), 0)
  private def loadKey(node: Code[Long], i: Int): Code[Long] = eltType.loadField(elementsType.loadField(elements(node), i), 0)

  private def childOffset(node: Code[Long], i: Int): Code[Long] =
    if (i == -1)
      storageType.fieldOffset(node, 1)
    else
      eltType.fieldOffset(elementsType.loadField(elements(node), i), 1)
  private def loadChild(node: Code[Long], i: Int): Code[Long] =
    Region.loadAddress(childOffset(node, i))
  private def setChild(parent: Code[Long], i: Int, child: Code[Long]): Code[Unit] =
    Code(
      if (i == -1) storageType.setFieldPresent(parent, 1) else Code._empty,
      Region.storeAddress(childOffset(parent, i), child),
      storageType.setFieldPresent(child, 0),
      Region.storeAddress(storageType.fieldOffset(child, 0), parent))

  private val insert: EmitMethodBuilder = {
    val insertAt = fb.newMethod("insert", Array[TypeInfo[_]](typeInfo[Long], typeInfo[Int], typeInfo[Boolean], typeToTypeInfo(key.compType), typeInfo[Long]), typeInfo[Long])
    val node: Code[Long] = insertAt.getArg[Long](1)
    val insertIdx: Code[Int] = insertAt.getArg[Int](2)
    val km: Code[Boolean] = insertAt.getArg[Boolean](3)
    val kv: Code[_] = insertAt.getArg(4)(typeToTypeInfo(key.compType))
    val child: Code[Long] = insertAt.getArg[Long](5)
    val parent = getParent(node)

    val newNode = insertAt.newLocal[Long]

    def makeUninitialized(idx: Int): Code[Long] =
      Code(
        setKeyPresent(node, idx),
        key.initializeEmpty(keyOffset(node, idx)),
        (!isLeaf(node)).orEmpty(setChild(node, idx, child)),
        loadKey(node, idx))

    def copyFrom(destNode: Code[Long], destIdx: Int, srcNode: Code[Long], srcIdx: Int): Code[Unit] =
      Code(
        setKeyPresent(destNode, destIdx),
        key.copy(keyOffset(srcNode, srcIdx), keyOffset(destNode, destIdx)),
        (!isLeaf(srcNode)).orEmpty(setChild(destNode, destIdx, loadChild(srcNode, srcIdx))))

    val shiftAndInsert = Array.range(1, maxElements)
      .foldLeft(makeUninitialized(0)) { (cont, destIdx) =>
        Code(
          hasKey(node, destIdx - 1).orEmpty(
            copyFrom(node, destIdx, node, destIdx - 1)),
          insertIdx.ceq(destIdx).mux(
            makeUninitialized(destIdx),
            cont))
      }

    def copyToNew(startIdx: Int) = coerce[Unit](Code(
      Array.tabulate(maxElements - startIdx) { newIdx =>
        Code(copyFrom(newNode, newIdx, node, newIdx + startIdx),
          setKeyMissing(node, newIdx + startIdx))
      }: _*))

    def insertKey(m: Code[Boolean], v: Code[_], c: Code[Long]): Code[Long] = {
      val upperBound = Array.range(0, maxElements)
        .foldRight(const(maxElements)) { (i, cont) =>
          (!hasKey(parent, i) ||
            key.compWithKey(loadKey(parent, i), m -> v) >= 0)
            .mux(i, cont)
        }
      Code((!isLeaf(node)).orEmpty(
        setChild(newNode, -1, c)),
        insertAt.invoke[Long](parent, upperBound, m, v, newNode))
    }

    def promote(idx: Int) = {
      val upperBound = Array.range(0, maxElements)
        .foldRight(const(maxElements)) { (i, cont) =>
          (!hasKey(parent, i) ||
            key.compSame(loadKey(parent, i), loadKey(node, idx)) >= 0)
            .mux(i, cont)
        }
      val (compKeyM, compKeyV) = key.loadCompKey(loadKey(node, idx))
      Code((!isLeaf(node)).orEmpty(
        setChild(newNode, -1, loadChild(node, idx))),
        key.copy(loadKey(node, idx),
          insertAt.invoke[Long](parent, upperBound, compKeyM, compKeyV, newNode)),
        setKeyMissing(node, idx))
    }

    val splitAndInsert = Code(
      isRoot(node).orEmpty(Code(
        createNode(root),
        setChild(root, -1, node))),
      createNode(newNode),
      (insertIdx > splitIdx).mux(
        Code(copyToNew(splitIdx + 1),
          promote(splitIdx),
          insertAt.invoke(newNode, insertIdx - splitIdx - 1, km, kv, child)),
        Code(
          copyToNew(splitIdx),
          insertIdx.ceq(splitIdx).mux(
            insertKey(km, kv, child),
            Code(promote(splitIdx - 1),
              insertAt.invoke(node, insertIdx, km, kv, child))))))

    insertAt.emit(isFull(node).mux(splitAndInsert, shiftAndInsert))
    insertAt
  }

  private val getF: EmitMethodBuilder = {
    val get = fb.newMethod("get", Array[TypeInfo[_]](typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(key.compType)), typeInfo[Long])
    val node = get.getArg[Long](1).load()
    val km = get.getArg[Boolean](2).load()
    val kv = get.getArg(3)(typeToTypeInfo(key.compType)).load()

    val cmp = get.newLocal[Int]
    def eltEqual(i: Int) = hasKey(node, i) && cmp.ceq(0)
    def insertOrGetAt(i: Int) = isLeaf(node).mux(
        insert.invoke[Long](node, i, km, kv, 0L),
      get.invoke(loadChild(node, i - 1), km, kv))

    get.emit(Code(
      cmp := 0,
      Array.range(0, maxElements).foldRight(insertOrGetAt(maxElements)) { (i, cont) =>
        val storeCmp = cmp := key.compWithKey(loadKey(node, i), km -> kv)
        (!hasKey(node, i) || Code(storeCmp, cmp >= 0)).mux(
          eltEqual(i).mux(loadKey(node, i),
            insertOrGetAt(i)),
          cont)
      }))
    get
  }

  def init: Code[Unit] = createNode(root)

  def getOrElseInitialize(km: Code[Boolean], kv: Code[_]): Code[Long] =
    getF.invoke(root, km, kv)

  def foreach(visitor: Code[Long] => Code[Unit]): Code[Unit] = {
    val f = fb.newMethod("foreach", Array[TypeInfo[_]](typeInfo[Long]), typeInfo[Unit])
    val node = f.getArg[Long](1)

    f.emit(Code(
      (!isLeaf(node)).orEmpty(f.invoke(loadChild(node, -1))),
      Array.range(0, maxElements)
        .foldRight(Code._empty[Unit]) { (i, cont) =>
          hasKey(node, i).orEmpty(
            Code(
              visitor(loadKey(node, i)),
              (!isLeaf(node)).orEmpty(f.invoke(loadChild(node, i))),
              cont))
        }))
    f.invoke(root)
  }

  val deepCopy: Code[Long] => Code[Unit] = {
    val f = fb.newMethod("deepCopy", Array[TypeInfo[_]](typeInfo[Long], typeInfo[Long]), typeInfo[Unit])
    val destNode = f.getArg[Long](1)
    val srcNode = f.getArg[Long](2)

    val er = EmitRegion(f, region)
    val newNode = f.newLocal[Long]

    def copyChild(i: Int) =
      Code(createNode(newNode),
        f.invoke[Unit](newNode, loadChild(srcNode, i)))

    val copyNodes = Array.range(0, maxElements).foldRight(Code._empty[Unit]) { (i, cont) =>
      hasKey(srcNode, i).orEmpty(
        Code(
          key.deepCopy(er, destNode, srcNode),
          (!isLeaf(srcNode)).orEmpty(Code(
            copyChild(i),
            setChild(destNode, i, newNode))),
          cont))
    }

    f.emit(Code(
      (!isLeaf(srcNode)).orEmpty(
        Code(copyChild(-1),
          setChild(destNode, -1, newNode))),
      copyNodes))

    { srcRoot: Code[Long] => f.invoke(root, srcRoot) }
  }

  def bulkStore(ob: Code[OutputBuffer])(keyStore: (Code[OutputBuffer], Code[Long]) => Code[Unit]): Code[Unit] = {
    val f = fb.newMethod("bulkStore", Array[TypeInfo[_]](typeInfo[Long]), typeInfo[Unit])
    val node = f.getArg[Long](1)

    f.emit(Code(
      ob.writeBoolean(!isLeaf(node)),
      (!isLeaf(node)).orEmpty(f.invoke(loadChild(node, -1))),
      Array.range(0, maxElements).foldRight(Code._empty[Unit]) { (i, cont) =>
        hasKey(node, i).mux(Code(
          ob.writeBoolean(true),
          keyStore(ob, loadKey(node, i)),
          (!isLeaf(node)).orEmpty(f.invoke(loadChild(node, i))),
          cont),
          ob.writeBoolean(false)) }))
    f.invoke(root)
  }

  def bulkLoad(ib: Code[InputBuffer])(keyLoad: (Code[InputBuffer], Code[Long]) => Code[Unit]): Code[Unit] = {
    val f = fb.newMethod("bulkLoad", Array[TypeInfo[_]](typeInfo[Long]), typeInfo[Unit])
    val node = f.getArg[Long](1)
    val newNode = f.newLocal[Long]
    val isInternalNode = f.newLocal[Boolean]

    f.emit(Code(
      isInternalNode := ib.readBoolean(),
      isInternalNode.orEmpty(
        Code(
          createNode(newNode),
          setChild(node, -1, newNode),
          f.invoke(newNode)
      )),
      Array.range(0, maxElements).foldRight(Code._empty[Unit]) { (i, cont) =>
        ib.readBoolean().orEmpty(Code(
          setKeyPresent(node, i),
          keyLoad(ib, keyOffset(node, i)),
          isInternalNode.orEmpty(
            Code(createNode(newNode),
            setChild(node, i, newNode),
            f.invoke(newNode))),
          cont))
      }))
    f.invoke(root)
  }
}

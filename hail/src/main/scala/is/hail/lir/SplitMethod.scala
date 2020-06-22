package is.hail.lir

import java.util.Comparator

import is.hail.asm4s._
import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm._

import scala.collection.mutable

class SplitReturn() extends Exception()

object SplitMethod {
  val TargetMethodSize: Int = 2000

  def apply(
    c: Classx[_],
    m: Method,
    blocks: Array[Block],
    pst: PST
  ): Classx[_] = {
    val split = new SplitMethod(c, m, blocks, pst)
    split.split()
    split.spillsClass
  }
}

class SplitMethod(
  c: Classx[_],
  m: Method,
  blocks: Array[Block],
  pst: PST
) {
  def nBlocks: Int = blocks.length

  val blockPartitions = new UnionFind(nBlocks)
  (0 until nBlocks).foreach { i => blockPartitions.makeSet(i) }

  private var methodSize = 0
  private val regionSize = new Array[Int](pst.nRegions)

  // compute methodSize and regionSize
  private def computeSizes(): Unit = {
    val blockSize = new Array[Int](nBlocks)

    var i = 0
    while (i < nBlocks) {
      val size = blocks(i).approxByteCodeSize()
      blockSize(i) = size
      methodSize += size
      i += 1
    }

    i = pst.nRegions - 1
    while (i >= 0) {
      val r = pst.regions(i)

      var c = 0
      var ci = 0
      var child: Region = null

      if (c < r.children.length) {
        ci = r.children(c)
        assert(ci > i)
        child = pst.regions(ci)
      }

      var size = 0
      var j = r.start
      while (j <= r.end) {
        if (child != null && j == child.start) {
          // children can overlap, don't double-count
          size += regionSize(ci) - blockSize(child.end)
          j = child.end
          c += 1
          if (c < r.children.length) {
            ci = r.children(c)
            assert(ci > i)
            child = pst.regions(ci)
          }
        } else {
          assert(child == null || j < child.start)
          size += blockSize(j)
          j += 1
        }
      }
      regionSize(i) = size

      i -= 1
    }
  }

  private val spillsClass = new Classx(genName("C", s"${ m.name }Spills"), "java/lang/Object")
  private val spillsCtor = {
    val ctor = spillsClass.newMethod("<init>", FastIndexedSeq(), UnitInfo)
    val L = new Block()
    ctor.setEntry(L)
    L.append(
      methodStmt(INVOKESPECIAL,
        "java/lang/Object",
        "<init>",
        "()V",
        false,
        UnitInfo,
        FastIndexedSeq(load(ctor.getParam(0)))))
    L.append(returnx())
    ctor
  }

  // index before creating spills
  private val (locals, localIdx) = m.findAndIndexLocals(blocks)

  private val spills = m.newLocal("spills", spillsClass.ti)

  private val paramFields = m.parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
    spillsClass.newField(genName("f", s"arg${ i + 1 }"), ti)
  }

  private val fields = locals.map { l =>
    if (l.isInstanceOf[Parameter])
      null
    else
      spillsClass.newField(genName("f", l.name), l.ti)
  }

  private val splitMethods = mutable.ArrayBuffer[Method]()

  def spillLocals(method: Method): Unit = {
    def localField(l: Local): Field =
      l match {
        case p: Parameter =>
          if (p.i == 0)
            null
          else {
            if (method eq m)
              null
            else
              paramFields(p.i - 1)
          }
        case _ =>
          localIdx.get(l) match {
            case Some(i) =>
              fields(i)
            case None =>
              null
          }
      }

    def getSpills(): ValueX = {
      if (method eq m)
        load(spills)
      else
        load(new Parameter(method, 1, spillsClass.ti))
    }

    def spill(x: X): Unit = {
      x.children.foreach(spill)
      x match {
        case x: LoadX =>
          val f = localField(x.l)
          if (f != null)
            x.replace(getField(f, getSpills()))
        case x: IincX =>
          val f = localField(x.l)
          if (f != null) {
            x.replace(
              putField(f, getSpills(),
                insn(IADD,
                  getField(f, getSpills()),
                  ldcInsn(x.i, IntInfo))))
          }
        case x: StoreX =>
          val f = localField(x.l)
          if (f != null) {
            val v = x.children(0)
            v.remove()
            x.replace(putField(f, getSpills(), v))
          }
        case _ =>
      }
    }

    for (b <- blocks) {
      var x = b.first
      while (x != null) {
        val n = x.next
        spill(x)
        x = n
      }
    }
  }

  def spillLocals(): Unit = {
    for (splitM <- splitMethods) {
      spillLocals(splitM)
    }
    spillLocals(m)
  }

  val q = new java.util.TreeSet[Integer](
    new Comparator[Integer] {
      def compare(i: Integer, j: Integer): Int =
        Integer.compare(regionSize(i), regionSize(j))
    })

  private var counter = 0
  private def genSplitMethodName(): String = {
    val c = counter
    counter += 1
    s"${ m.name }split$c"
  }

  private var spillReturnValue: Field = _

  private def splitRegion(i: Int): Unit = {
    val r = pst.regions(i)

    val Lstart = blocks(pst.linearization(r.start))
    val Lend = blocks(pst.linearization(r.end))

    val returnTI = Lend.last match {
      case _: GotoX => UnitInfo
      case _: IfX => BooleanInfo
      case _: SwitchX => IntInfo
      case _: ReturnX => m.returnTypeInfo
      case _: ThrowX => UnitInfo
    }

    val splitM = c.newMethod(genSplitMethodName(), FastIndexedSeq(spillsClass.ti), returnTI)
    splitMethods += splitM

    (r.start to r.end).foreach { i =>
      val b = blockPartitions.find(pst.linearization(i))
      val L = blocks(b)

      blocks(b) = null
      L.method = splitM

      // handle split return statements
      if (i < r.end) {
        L.last match {
          case x: ReturnX =>
            x.remove()
            if (m.catchSplitReturn == null) {
              if (m.returnTypeInfo == UnitInfo) {
                m.catchSplitReturn = returnx()
              } else {
                spillReturnValue = spillsClass.newField("returnValue", m.returnTypeInfo)
                m.catchSplitReturn = returnx(getField(spillReturnValue, load(spills)))
              }
            }
            if (m.returnTypeInfo != UnitInfo) {
              val v = x.children(0)
              v.remove()
              x.insertBefore(putField(spillReturnValue, load(spills), v))
            }
            val ti = classInfo[SplitReturn]
            val tcls = classOf[SplitReturn]
            val c = tcls.getDeclaredConstructor()
            x.replace(
              throwx(newInstance(ti,
                  Type.getInternalName(tcls), "<init>", Type.getConstructorDescriptor(c), ti, FastIndexedSeq())))
        }
      }
    }

    splitM.setEntry(Lstart)

    // replacement block for region
    val L = new Block()
    L.method = m
    Lstart.replace(L)

    blockPartitions.union(
      pst.linearization(r.start), pst.linearization(r.end))
    blocks(blockPartitions.find(pst.linearization(r.start))) = L

    def invokeSplitM(): ValueX =
      methodInsn(INVOKEVIRTUAL, splitM, Array(load(m.getParam(0)), load(spills)))

    def invokeSplitMStmt(): StmtX =
      methodStmt(INVOKEVIRTUAL, splitM, Array(load(m.getParam(0)), load(spills)))

    Lend.last match {
      case x: GotoX =>
        x.remove()
        Lend.append(returnx())
        L.append(invokeSplitMStmt())
        L.append(x)
      case x: ThrowX =>
        L.append(invokeSplitMStmt())
      case x: IfX =>
        val Ltrue = x.Ltrue
        val Lfalse = x.Lfalse

        val newLtrue = new Block()
        newLtrue.method = splitM
        newLtrue.append(returnx(ldcInsn(1, BooleanInfo)))
        x.setLtrue(newLtrue)

        val newLfalse = new Block()
        newLfalse.method = splitM
        newLfalse.append(returnx(ldcInsn(0, BooleanInfo)))
        x.setLfalse(newLfalse)

        L.append(
          ifx(IFNE, invokeSplitM(), Ltrue, Lfalse))
      case x: SwitchX => IntInfo
        x.remove()
        val i = x.children(0)
        i.remove()
        Lend.append(returnx(i))
        x.setChild(0, invokeSplitM())
        L.append(x)
      case _: ReturnX =>
        if (returnTI == UnitInfo) {
          L.append(invokeSplitMStmt())
          L.append(returnx())
        } else {
          L.append(returnx(invokeSplitM()))
        }
    }

    // update sizes
    val size = regionSize(i)

    methodSize -= size

    val p = pst.regions(i).parent
    while (p != -1) {
      q.remove(p)
      regionSize(p) -= size
      assert(pst.regions(p).children.nonEmpty)
      if (regionSize(p) < SplitMethod.TargetMethodSize) {
        q.add(p)
      }
    }
  }

  def splitRegions(): Unit = {
    var i = 0
    while (i < pst.nRegions) {
      if (regionSize(i) < SplitMethod.TargetMethodSize ||
        pst.regions(i).children.isEmpty) {
        q.add(i)
      }
      i += 1
    }

    while (methodSize > SplitMethod.TargetMethodSize &&
      !q.isEmpty) {
      val i = q.pollLast()
      splitRegion(i)
    }
  }

  def split(): Unit = {
    computeSizes()
    splitRegions()

    spillLocals()

    var x: StmtX = store(spills, new NewInstanceX(spillsClass.ti, spillsCtor))
    m.entry.prepend(x)

    // spill parameters
    m.parameterTypeInfo.indices.foreach { i =>
      val putParam = putField(
        paramFields(i),
        load(spills),
        load(m.getParam(i + 1)))
      x.insertAfter(putParam)
      x = putParam
    }
  }
}

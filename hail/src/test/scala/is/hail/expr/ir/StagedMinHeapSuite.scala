package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq}
import is.hail.asm4s.{ThisFieldRef, _}
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr.ir.functions.LocusFunctions
import is.hail.expr.ir.streams.EmitMinHeap
import is.hail.types.physical.stypes.concrete.SIndexablePointerValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus, PInt32}
import is.hail.utils.{FastSeq, using}
import is.hail.variant.{Locus, ReferenceGenome}
import org.scalatest.Matchers.{be, convertToAnyShouldWrapper}
import org.testng.annotations.Test

import scala.language.implicitConversions

class StagedMinHeapSuite extends HailSuite {

  @Test def testSorting(): Unit =
    forAll { (xs: IndexedSeq[Int]) => sort(xs) == xs.sorted }.check()

  @Test def testHeapProperty(): Unit =
    forAll { (xs: IndexedSeq[Int]) =>
      val heap = heapify(xs)
      (0 until heap.size / 2).forall { i =>
        ((2 * i + 1) >= heap.size || heap(i) <= heap(2 * i + 1)) &&
          ((2 * i + 2) >= heap.size || heap(i) <= heap(2 * i + 2))
      }
    }.check()

  @Test def testNonEmpty(): Unit =
    gen("NonEmpty") { (heap: IntHeap) =>
      heap.nonEmpty should be (false)
      for (i <- 0 to 10) heap.push(i)
      heap.nonEmpty should be (true)
      for (_ <- 0 to 10) heap.pop()
      heap.nonEmpty should be (false)
    }(BuildIntHeap)

  @Test def testInnerClass(): Unit =
    forAll(loci) { case (rg: ReferenceGenome, loci: IndexedSeq[Locus]) =>
      withReferenceGenome(rg) {

        val sortedLoci =
          gen("InnerClass") { (heap: LocusHeap) =>
            loci.foreach(heap.push)
            IndexedSeq.fill(loci.size)(heap.pop())
          }(buildLocusHeap(rg))

        sortedLoci == loci.sorted(rg.locusOrdering)
      }
    }.check()

  val loci: Gen[(ReferenceGenome, IndexedSeq[Locus])] =
    for {
      genome <- ReferenceGenome.gen
      loci <- Gen.buildableOf(Locus.gen(genome))
    } yield (genome, loci)

  def withReferenceGenome[A](rg: ReferenceGenome)(f: => A): A = {
    ctx.backend.addReference(rg)
    try { f } finally { ctx.backend.removeReference(rg.name) }
  }

  def sort(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    gen("Sort") { (heap: IntHeap) =>
      xs.foreach(heap.push)
      IndexedSeq.fill(xs.size)(heap.pop())
    }(BuildIntHeap)

  def heapify(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    gen("Heapify") { (heap: IntHeap) =>
      pool.scopedRegion { r =>
        xs.foreach(heap.push)
        val ptr = heap.toArray(r)
        SafeIndexedSeq(PCanonicalArray(PInt32()), ptr).asInstanceOf[IndexedSeq[Int]]
      }
    }(BuildIntHeap)

  def gen[Heap, A](name: String)(f: Heap => A)(implicit B: Build[Heap]): A = {
    val emodb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val Main = emodb.genEmitClass[B.Heap](name)(B.hti)

    def compare: EmitMethodBuilder[_] =
      Main.defineEmitMethod("compare", Array.fill(2)(B.sType.paramType), IntInfo) { mb =>
        mb.emitWithBuilder[Int] { cb =>
          val p = mb.getSCodeParam(1)
          val q = mb.getSCodeParam(2)
          Main.getOrdering(B.sType, B.sType).compareNonnull(cb, p, q)
        }
      }

    // The reference genome is added to the "Main" class by `resultWithIndex` and is not
    // accessible from the MinHeap. Thus, we need to define a comparator in the outer class
    // (Main) and hold a reference to it in the generated MinHeap.
    val MinHeap = EmitMinHeap(Main.emodb, B.sType) { classBuilder =>
      new EmitMinHeap.StagedComparator {
        val mainRef: ThisFieldRef[_] =
          classBuilder.genFieldThisRef("parent")(Main.cb.ti)

        override def init(cb: EmitCodeBuilder, enclosingRef: Value[AnyRef]): Unit =
          cb.assignAny(mainRef, Code.checkcast(enclosingRef)(Main.cb.ti))

        override def apply(cb: EmitCodeBuilder, a: SValue, b: SValue): Value[Int] =
          cb.invokeCode[Int](compare, mainRef, a, b)
      }
    }(Main)

    Main.defineEmitMethod("push", FastSeq(B.ti), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.push(cb, B.fromType(cb, Main.partitionRegion, mb.getCodeParam[B.Type](1)(B.ti)))
      }
    }

    Main.defineEmitMethod("pop", FastSeq(), B.ti) { mb =>
      mb.emitWithBuilder[B.Type] { cb =>
        val res = B.toType(cb, MinHeap.peek(cb))
        MinHeap.pop(cb)
        MinHeap.realloc(cb)
        res
      }
    }

    Main.defineEmitMethod("nonEmpty", FastSeq(), BooleanInfo) { mb =>
      mb.emitWithBuilder[Boolean] { MinHeap.nonEmpty }
    }

    Main.defineEmitMethod("toArray", FastSeq(typeInfo[Region]), LongInfo) { mb =>
      mb.emitWithBuilder { cb =>
        val region = mb.getCodeParam[Region](1)
        val arr = MinHeap.toArray(cb, region)
        arr.asInstanceOf[SIndexablePointerValue].a
      }
    }

    trait Resource extends AutoCloseable { def init(): Unit }

    Main.cb.addInterface(implicitly[TypeInfo[Resource]].iname)
    Main.defineEmitMethod("init", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        // Properties like pool and reference genomes are set after `Main`'s
        // default constructor is called, thus we need a separate method to
        // initialise the heap with them.
        MinHeap.init(cb, Main.pool())
      }
    }
    Main.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { MinHeap.close }
    }

    pool.scopedRegion { r =>
      val heap = Main
        .resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)
        .asInstanceOf[B.Heap with Resource]

      heap.init()
      using(heap)(f)
    }
  }

  trait LocusHeap extends Heap {
    def push(locus: Locus): Unit
    def pop(): Locus
  }

  trait IntHeap extends Heap {
    def push(x: Int): Unit
    def pop(): Int
  }

  trait Heap {
    def nonEmpty: Boolean
    def toArray(r: Region): Long
  }

  sealed trait Build[H] {
    type Heap = H
    type Type

    def hti: TypeInfo[Heap]
    def ti: TypeInfo[Type]

    def sType: SType
    def fromType(cb: EmitCodeBuilder, region: Value[Region], a: Value[Type]): SValue
    def toType(cb: EmitCodeBuilder, sa: SValue): Value[Type]
  }

  implicit object BuildIntHeap extends Build[IntHeap] {
    override type Type =
      Int
    override def hti: TypeInfo[IntHeap] =
      implicitly
    override def ti: TypeInfo[Int] =
      implicitly

    override def sType: SType =
      SInt32
    override def fromType(cb: EmitCodeBuilder, region: Value[Region], a: Value[Int]): SValue =
      new SInt32Value(a)
    override def toType(cb: EmitCodeBuilder, sa: SValue): Value[Int] =
      sa.asInt.value
  }

  implicit def buildLocusHeap(rg: ReferenceGenome): Build[LocusHeap]  =
    new Build[LocusHeap] {
      override type Type =
        Locus
      override def hti: TypeInfo[LocusHeap] =
        implicitly
      override def ti: TypeInfo[Locus] =
        implicitly

      override def sType: SType =
        PCanonicalLocus(rg.name, required = true).sType
      override def fromType(cb: EmitCodeBuilder, region: Value[Region], a: Value[Locus]): SValue =
        LocusFunctions.emitLocus(cb, region, a, sType.storageType().asInstanceOf[PCanonicalLocus])
      override def toType(cb: EmitCodeBuilder, sa: SValue): Value[Locus] =
        sa.asLocus.getLocusObj(cb)
    }
}

package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr.ir.functions.LocusFunctions
import is.hail.expr.ir.streams.StagedMinHeap
import is.hail.types.physical.stypes.concrete.SIndexablePointerValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus, PInt32Required}
import is.hail.utils.{FastSeq, using}
import is.hail.variant.{Locus, ReferenceGenome}
import org.scalatest.Matchers.{be, convertToAnyShouldWrapper}
import org.testng.annotations.Test

import scala.language.{higherKinds, implicitConversions}

sealed trait StagedCoercions[A] {
  def ti: TypeInfo[A]
  def sType: SType
  def fromType(cb: EmitCodeBuilder, region: Value[Region], a: Value[A]): SValue
  def toType(cb: EmitCodeBuilder, sa: SValue): Value[A]
}

sealed trait StagedCoercionInstances {
  implicit object StagedIntCoercions extends StagedCoercions[Int] {
    override def ti: TypeInfo[Int] =
      implicitly
    override def sType: SType =
      SInt32
    override def fromType(cb: EmitCodeBuilder, region: Value[Region], a: Value[Int]): SValue =
      new SInt32Value(a)
    override def toType(cb: EmitCodeBuilder, sa: SValue): Value[Int] =
      sa.asInt.value
  }

  def stagedLocusCoercions(rg: ReferenceGenome): StagedCoercions[Locus] =
    new StagedCoercions[Locus] {
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

class StagedMinHeapSuite extends HailSuite with StagedCoercionInstances {

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
    }

  val loci: Gen[(ReferenceGenome, IndexedSeq[Locus])] =
    for {
      genome <- ReferenceGenome.gen
      loci <- Gen.buildableOf(Locus.gen(genome))
    } yield (genome, loci)

  @Test def testLocus(): Unit =
    forAll(loci) { case (rg: ReferenceGenome, loci: IndexedSeq[Locus]) =>
      withReferenceGenome(rg) {

        val sortedLoci =
          gen("Locus", stagedLocusCoercions(rg)) { (heap: LocusHeap) =>
            loci.foreach(heap.push)
            IndexedSeq.fill(loci.size)(heap.pop())
          }

        sortedLoci == loci.sorted(rg.locusOrdering)
      }
    }.check()

  def withReferenceGenome[A](rg: ReferenceGenome)(f: => A): A = {
    ctx.backend.addReference(rg)
    try { f } finally { ctx.backend.removeReference(rg.name) }
  }

  def sort(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    gen("Sort") { (heap: IntHeap) =>
      xs.foreach(heap.push)
      IndexedSeq.fill(xs.size)(heap.pop())
    }

  def heapify(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    gen("Heapify") { (heap: IntHeap) =>
      pool.scopedRegion { r =>
        xs.foreach(heap.push)
        val ptr = heap.toArray(r)
        SafeIndexedSeq(PCanonicalArray(PInt32Required), ptr).asInstanceOf[IndexedSeq[Int]]
      }
    }

  def gen[H <: Heap[A], A, B](name: String, A: StagedCoercions[A])(f: H => B)(implicit H: TypeInfo[H]): B =
    gen[H, A, B](name)(f)(H, A)

  def gen[H <: Heap[A], A, B](name: String)(f: H => B)(implicit H: TypeInfo[H], A: StagedCoercions[A]): B = {
    val emodb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val Main = emodb.genEmitClass[H](name)(H)

    val MinHeap = StagedMinHeap(Main.emodb, A.sType) {
      (cb: EmitCodeBuilder, x: SValue, y: SValue) =>
        cb.emb.ecb.getOrdering(A.sType, A.sType).compareNonnull(cb, x, y)
    }(Main)

    Main.defineEmitMethod("push", FastSeq(A.ti), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.push(cb, A.fromType(cb, Main.partitionRegion, mb.getCodeParam[A](1)(A.ti)))
      }
    }

    Main.defineEmitMethod("pop", FastSeq(), A.ti) { mb =>
      mb.emitWithBuilder[A] { cb =>
        val res = A.toType(cb, MinHeap.peek(cb))
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
        .asInstanceOf[H with Resource]

      heap.init()
      using(heap)(f)
    }
  }

  trait LocusHeap extends Heap[Locus] {
    def push(locus: Locus): Unit
    def pop(): Locus
  }

  trait IntHeap extends Heap[Int] {
    def push(x: Int): Unit
    def pop(): Int
  }

  trait Heap[A] {
    def nonEmpty: Boolean
    def toArray(r: Region): Long
  }
}


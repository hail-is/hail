package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr.ir.functions.LocusFunctions
import is.hail.expr.ir.streams.{EmitMinHeap, StagedMinHeap}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SIndexablePointerValue
import is.hail.types.physical.stypes.primitives.SInt32
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus, PInt32}
import is.hail.utils.{FastSeq, using}
import is.hail.variant.{Locus, ReferenceGenome}
import org.scalatest.Matchers.{be, convertToAnyShouldWrapper}
import org.testng.annotations.Test

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
    intHeap("NonEmpty") { heap =>
      heap.nonEmpty should be (false)
      for (i <- 0 to 10) heap.push(i)
      heap.nonEmpty should be (true)
      for (_ <- 0 to 10) heap.pop()
      heap.nonEmpty should be (false)
    }

  @Test def testClosure(): Unit =
    forAll(loci) { case (rg: ReferenceGenome, loci: IndexedSeq[Locus]) =>
      withGenome(rg) {
        val emodb = new EmitModuleBuilder(ctx, new ModuleBuilder())
        val Main = emodb.newEmitClass[LocusHeap]("Closure")

        val eltPTy: PCanonicalLocus =
          PCanonicalLocus(rg.name, required = true)

        val compare: EmitMethodBuilder[_] =
          Main.defineEmitMethod("compare", Array.fill(2)(eltPTy.sType.paramType), IntInfo) { mb =>
            mb.emitWithBuilder[Int] { cb =>
              val p = mb.getSCodeParam(1)
              val q = mb.getSCodeParam(2)
              Main.getOrdering(eltPTy.sType, eltPTy.sType).compareNonnull(cb, p, q)
            }
          }

        // The reference genome is added to the "Main" class by `resultWithIndex` and is not
        // accessible from the MinHeap. Thus, we need to define a comparator in the outer class
        // (Main) and hold a reference to it in the generated MinHeap.
        val MinHeap = EmitMinHeap(Main.emodb, eltPTy.sType) { classBuilder =>
          new EmitMinHeap.StagedComparator {
            val mainRef: ThisFieldRef[_] =
              classBuilder.genFieldThisRef("parent")(Main.cb.ti)

            override def init(cb: EmitCodeBuilder, enclosingRef: Value[AnyRef]): Unit =
              cb.assignAny(mainRef, Code.checkcast(enclosingRef)(Main.cb.ti))

            override def apply(cb: EmitCodeBuilder, a: SValue, b: SValue): Value[Int] =
              cb.invokeCode[Int](compare, mainRef, a, b)
          }
        }(Main)

        Main.defineEmitMethod("push", FastSeq(typeInfo[Locus]), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            val locus = mb.getCodeParam[Locus](1)
            val sLocus = LocusFunctions.emitLocus(cb, Main.partitionRegion, locus, eltPTy)
            MinHeap.push(cb, sLocus)
          }
        }

        Main.defineEmitMethod("pop", FastSeq(), typeInfo[Locus]) { mb =>
          mb.emitWithBuilder[Locus] { cb =>
            val res = MinHeap.peek(cb).asLocus.getLocusObj(cb)
            MinHeap.pop(cb)
            MinHeap.realloc(cb)
            res
          }
        }

        val sortedLoci =
          Heap.implementAndUse(Main, MinHeap) { heap =>
            loci.foreach(heap.push)
            IndexedSeq.fill(loci.size)(heap.pop())
          }

        sortedLoci == loci.sorted(rg.locusOrdering)
      }
    }.check()

  val loci: Gen[(ReferenceGenome, IndexedSeq[Locus])] =
    for {
      genome <- ReferenceGenome.gen
      loci <- Gen.buildableOf(Locus.gen(genome))
    } yield (genome, loci)

  def withGenome[A](rg: ReferenceGenome)(f: => A): A = {
    ctx.backend.addReference(rg)
    try { f } finally { ctx.backend.removeReference(rg.name) }
  }

  def sort(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    intHeap("Sort") { heap =>
      xs.foreach(heap.push)
      IndexedSeq.fill(xs.size)(heap.pop())
    }

  def heapify(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    intHeap("Heapify") { heap =>
      pool.scopedRegion { r =>
        xs.foreach(heap.push)
        val ptr = heap.toArray(r)
        SafeIndexedSeq(PCanonicalArray(PInt32()), ptr).asInstanceOf[IndexedSeq[Int]]
      }
    }

  def intHeap[B](name: String)(f: IntHeap => B): B = {
    val emodb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val Main = emodb.newEmitClass[IntHeap](name)
    val MinHeap = EmitMinHeap(Main.emodb, SInt32) { classBuilder =>
      (cb: EmitCodeBuilder, a: SValue, b: SValue) =>
        classBuilder.getOrdering(SInt32, SInt32).compareNonnull(cb, a, b)
    }(Main)

    Main.defineEmitMethod("push", FastSeq(SInt32.paramType), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.push(cb, mb.getSCodeParam(1))
      }
    }

    Main.defineEmitMethod("pop", FastSeq(), SInt32.paramType) { mb =>
      mb.emitSCode { cb =>
        val res = MinHeap.peek(cb)
        MinHeap.pop(cb)
        MinHeap.realloc(cb)
        res
      }
    }

    Heap.implementAndUse(Main, MinHeap) { f }
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

  object Heap {
    def implementAndUse[A <: Heap, B](Main: EmitClassBuilder[A], MinHeap: StagedMinHeap)
                                     (test: A => B)
    : B = {

      Main.defineEmitMethod("nonEmpty", FastSeq(), BooleanInfo) { mb =>
        mb.emitWithBuilder[Boolean] {
          MinHeap.nonEmpty
        }
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
        mb.voidWithBuilder {
          MinHeap.close
        }
      }

      pool.scopedRegion { r =>
        val heap = Main
          .resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)
          .asInstanceOf[A with Resource]

        heap.init()
        using(heap)(test)
      }
    }
  }
}

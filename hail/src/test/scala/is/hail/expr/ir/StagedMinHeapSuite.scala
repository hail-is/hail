package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq}
import is.hail.asm4s._
import is.hail.check.{Arbitrary, Gen}
import is.hail.check.Prop.forAll
import is.hail.expr.ir.streams.EmitMinHeap
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SIndexablePointerValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus, PInt32}
import is.hail.utils.{FastSeq, using}
import is.hail.variant.{Locus, ReferenceGenome}
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

  @Test def testClosure(): Unit =
    forAll(loci) { case (rg: ReferenceGenome, loci: IndexedSeq[Locus]) =>
      withGenome(rg) {
        trait LocusHeap extends Heap { def pop(): Locus }

        val modb = new EmitModuleBuilder(ctx, new ModuleBuilder())
        val Main = modb.genEmitClass[LocusHeap with AutoCloseable]("Closure")
        Main.cb.addInterface(implicitly[TypeInfo[AutoCloseable]].iname)

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
        // accessible from the MinHeap. Thus, we need a reference to the outer class (Main)
        // to dispatch
        val MinHeap = EmitMinHeap(modb, eltPTy.sType) { classBuilder =>
          new EmitMinHeap.StagedComparator {
            val parent: ThisFieldRef[_] =
              classBuilder.genFieldThisRef("parent")(Main.cb.ti)

            override def init(cb: EmitCodeBuilder, enclosingRef: Value[AnyRef]): Unit =
              cb.assignAny(parent, Code.checkcast(enclosingRef)(Main.cb.ti))

            override def apply(cb: EmitCodeBuilder, a: SValue, b: SValue): Value[Int] =
              cb.invokeCode[Int](compare, parent, a, b)
          }
        }(Main)

        Main.defineEmitMethod("init", FastSeq(), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            MinHeap.init(cb, Main.pool())
            for (x <- loci) {
              val slocus = eltPTy.constructFromContigAndPosition(cb, Main.partitionRegion, x.contig, x.position)
              MinHeap.push(cb, slocus)
            }
          }
        }

        Main.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
          mb.voidWithBuilder {
            MinHeap.close
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

        Main.defineEmitMethod("toArray", FastSeq(typeInfo[Region]), LongInfo) { mb =>
          mb.emitWithBuilder { cb =>
            val region = mb.getCodeParam[Region](1)
            val arr = MinHeap.toArray(cb, region)
            arr.asInstanceOf[SIndexablePointerValue].a
          }
        }

        val sortedLoci =
          pool.scopedRegion { r =>
            using(Main.resultWithIndex(ctx.shouldWriteIRFiles())(theHailClassLoader, ctx.fs, ctx.taskContext, r)) {
              heap =>
                heap.init()
                IndexedSeq.fill(loci.size)(heap.pop())
            }
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
    using(() => ctx.backend.removeReference(rg.name))(_ => f)
  }

  def sort(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    intHeap("Sort", xs) { heap =>
      heap.init()
      IndexedSeq.fill(xs.size)(heap.pop())
    }

  def heapify(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    intHeap("Heapify", xs) { heap =>
      pool.scopedRegion { r =>
        heap.init()
        val ptr = heap.toArray(r)
        SafeIndexedSeq(PCanonicalArray(PInt32()), ptr).asInstanceOf[IndexedSeq[Int]]
      }
    }

  def intHeap[B](name: String, xs: IndexedSeq[Int])(f: IntHeap => B): B = {
    val modb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val Main = modb.genEmitClass[IntHeap with AutoCloseable](name)
    Main.cb.addInterface(implicitly[TypeInfo[AutoCloseable]].iname)

    val MinHeap = EmitMinHeap(modb, SInt32) { _ =>
      (cb: EmitCodeBuilder, a: SValue, b: SValue) =>
        cb.memoize({
          val x = a.asPrimitive.primitiveValue[Int]
          val y = b.asPrimitive.primitiveValue[Int]
          Code.invokeStatic[Int](classOf[Integer], "compare", Array.fill(2)(classOf[Int]), Array(x, y))
        })
    }(Main)

    Main.defineEmitMethod("init", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.init(cb, Main.pool())
        for (x <- xs) MinHeap.push(cb, new SInt32Value(x))
      }
    }

    Main.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { MinHeap.close }
    }

    Main.defineEmitMethod("pop", FastSeq(), IntInfo) { mb =>
      mb.emitWithBuilder[Int] { cb =>
        val res = MinHeap.peek(cb)
        MinHeap.pop(cb)
        MinHeap.realloc(cb)
        res.asPrimitive.primitiveValue[Int]
      }
    }

    Main.defineEmitMethod("toArray", FastSeq(typeInfo[Region]), LongInfo) { mb =>
      mb.emitWithBuilder { cb =>
        val region = mb.getCodeParam[Region](1)
        val arr = MinHeap.toArray(cb, region)
        arr.asInstanceOf[SIndexablePointerValue].a
      }
    }

    pool.scopedRegion { r =>
      using(Main.resultWithIndex(ctx.shouldWriteIRFiles())(theHailClassLoader, ctx.fs, ctx.taskContext, r)) { f }
    }
  }

  trait IntHeap extends Heap { def pop(): Int }

  trait Heap {
    def init(): Unit
    def toArray(r: Region): Long
  }
}

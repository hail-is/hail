package is.hail.annotations

import is.hail.utils._
import is.hail.expr.ir.TypeToTypeInfo
import is.hail.expr._
import is.hail.asm4s._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

object InplaceSort {
  def insertionSort(region: Code[MemoryBuffer], fb: FunctionBuilder[_], aOff: Code[Long], t: TArray): Code[Unit] = {
    val i = fb.newLocal[Int]
    val j = fb.newLocal[Int]
    val x = fb.newLocal[Long]
    val y = fb.newLocal[Long]
    val ord = t.elementType.unsafeOrdering(true)
    Code(
      i := 1,
      Code.whileLoop(i < TContainer.loadLength(region, aOff),
        x := t.loadElement(region, aOff, i),
        j := i - 1,
        y := t.loadElement(region, aOff, j),
        Code.whileLoop(j > 0 && (ord.compare(region, x, region, y) < 0),
          region.copyFrom(region, y, t.loadElement(region, aOff, j + 1), t.elementType.byteSize),
          y := t.loadElement(region, aOff, j)
        ),
        region.copyFrom(region, x, t.loadElement(region, aOff, j + 1), t.elementType.byteSize),
        i := i + 1
      )
    )
  }

  def apply(region: Code[MemoryBuffer], fb: FunctionBuilder[_], aOff: Code[Long], t: TArray): Code[Unit] = {
    // (TContainer.loadLength(region, aOff) < 10).mux(
    insertionSort(region, fb, aOff, t)
    // ,
    //   Code.invokeStatic[RegionValueQuickSort, MemoryBuffer, Long, TArray]("quickSort", region, aOff, t)
    // )
  }
}

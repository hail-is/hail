package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.{RIterable, TypeWithRequiredness}
import is.hail.types.physical.{PCanonicalArray, PContainer}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.stypes.primitives.{SInt32Value, SInt64Value}

trait SContainer extends SType {
  def elementType: SType
  def elementEmitType: EmitType

  override def _typeWithRequiredness: TypeWithRequiredness =
    RIterable(elementEmitType.typeWithRequiredness.r)
}

trait SIndexableValue extends SValue {
  def st: SContainer

  def loadLength(): Value[Int]

  def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Value[Boolean]

  def isElementDefined(cb: EmitCodeBuilder, i: Code[Int]): Value[Boolean] =
    cb.memoize(!isElementMissing(cb, i))

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode

  def hasMissingValues(cb: EmitCodeBuilder): Value[Boolean]

  def castToArray(cb: EmitCodeBuilder): SIndexableValue

  def forEachDefined(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SValue) => Unit)
    : Unit = {
    val length = loadLength()
    val idx = cb.newLocal[Int]("foreach_idx", 0)
    cb.while_(
      idx < length, {

        loadElement(cb, idx).consume(
          cb,
          {}, /*do nothing if missing*/
          eltCode => f(cb, idx, eltCode),
        )
        cb.assign(idx, idx + 1)
      },
    )
  }

  def forEachDefinedOrMissing(
    cb: EmitCodeBuilder
  )(
    missingF: (EmitCodeBuilder, Value[Int]) => Unit,
    presentF: (EmitCodeBuilder, Value[Int], SValue) => Unit,
  ): Unit = {
    val length = loadLength()
    val idx = cb.newLocal[Int]("foreach_idx", 0)
    cb.while_(
      idx < length, {

        loadElement(cb, idx).consume(
          cb,
          /* do function if missing */
          missingF(cb, idx),
          eltCode => presentF(cb, idx, eltCode),
        )
        cb.assign(idx, idx + 1)
      },
    )
  }

  override def hash(cb: EmitCodeBuilder): SInt32Value = {
    val hash_result = cb.newLocal[Int]("array_hash", 1)
    forEachDefinedOrMissing(cb)(
      {
        case (cb, idx) => cb.assign(hash_result, hash_result * 31)
      },
      {
        case (cb, idx, element) =>
          cb.assign(hash_result, hash_result * 31 + element.hash(cb).value)
      },
    )
    new SInt32Value(hash_result)
  }

  def sliceArray(
    cb: EmitCodeBuilder,
    region: Value[Region],
    pt: PCanonicalArray,
    start: Code[Int],
    end: Code[Int],
    deepCopy: Boolean = false,
  ): SIndexableValue = {
    val startMemo = cb.newLocal[Int]("sindexable_slice_array_start_memo", start)
    pt.constructFromElements(
      cb,
      region,
      cb.newLocal[Int]("slice_length", end - startMemo),
      deepCopy,
    )((cb, idx) => this.loadElement(cb, idx + startMemo))
  }

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val storageType = this.st.storageType().asInstanceOf[PContainer]
    val length = this.loadLength()
    val totalSize =
      cb.newLocal[Long]("sindexableptr_size_in_bytes", storageType.elementsOffset(length).toL)
    if (this.st.elementType.containsPointers) {
      this.forEachDefined(cb) { (cb, _, element) =>
        cb.assign(totalSize, totalSize + element.sizeToStoreInBytes(cb).value)
      }
    } else {
      cb.assign(totalSize, storageType.contentsByteSize(length))
    }

    new SInt64Value(totalSize)
  }
}

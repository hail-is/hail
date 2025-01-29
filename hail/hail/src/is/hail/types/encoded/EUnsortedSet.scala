package is.hail.types.encoded

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{ArraySorter, EmitCodeBuilder, EmitRegion, StagedArrayBuilder}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue, SingleCodeType}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.virtual._
import is.hail.utils._

final case class EUnsortedSet(val elementType: EType, override val required: Boolean = false)
    extends EContainer {
  private[this] val arrayRepr = EArray(elementType, required)

  def _decodedSType(requestedType: Type): SType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    requestedType match {
      case _: TSet =>
        SIndexablePointer(PCanonicalSet(elementPType, false))
    }
  }

  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    // Anything we have to encode from a region should already be sorted so we don't
    // have to do anything else
    arrayRepr._buildEncoder(cb, v, out)

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer])
    : SValue = {
    val tmpRegion = cb.memoize(Region.stagedCreate(Region.REGULAR, region.getPool()), "tmp_region")

    val arrayDecoder = arrayRepr.buildDecoder(t, cb.emb.ecb)
    val decodedUnsortedArray = arrayDecoder(cb, region, in).asInstanceOf[SIndexablePointerValue]
    val sct = SingleCodeType.fromSType(decodedUnsortedArray.st.elementType)

    val ab = new StagedArrayBuilder(cb, sct, elementType.required, 0)
    ab.ensureCapacity(cb, decodedUnsortedArray.length)
    decodedUnsortedArray.forEachDefined(cb) { (cb, i, res) =>
      ab.add(cb, ab.elt.coerceSCode(cb, res, region, false).code)
    }

    val sorter = new ArraySorter(EmitRegion(cb.emb, region), ab)
    def lessThan(cb: EmitCodeBuilder, region: Value[Region], l: Value[_], r: Value[_])
      : Value[Boolean] =
      cb.emb.ecb.getOrdering(sct.loadedSType, sct.loadedSType)
        .ltNonnull(cb, sct.loadToSValue(cb, l), sct.loadToSValue(cb, r))

    sorter.sort(cb, tmpRegion, lessThan)
    /* TODO Should be able to overwrite the unsorted array with sorted contents instead of
     * allocating */
    val ret = sorter.toRegion(cb, t)
    cb.append(tmpRegion.invalidate())
    ret
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    arrayRepr._buildSkip(cb, r, in)

  def _asIdent = s"set_of_${elementType.asIdent}"
  def _toPretty = s"EUnsortedSet[$elementType]"
  def setRequired(newRequired: Boolean): EType = EUnsortedSet(elementType, newRequired)
}

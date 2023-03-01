package is.hail.annotations

import is.hail.types.physical.PType
import is.hail.backend.HailStateManager

object ScalaToRegionValue {
  def apply(sm: HailStateManager, region: Region, t: PType, a: Annotation): Long = {
    t.unstagedStoreJavaObject(sm, a, region)
  }
}

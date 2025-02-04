package is.hail.annotations

import is.hail.backend.HailStateManager
import is.hail.types.physical.PType

object ScalaToRegionValue {
  def apply(sm: HailStateManager, region: Region, t: PType, a: Annotation): Long =
    t.unstagedStoreJavaObject(sm, a, region)
}

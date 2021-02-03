package is.hail.annotations

import is.hail.types.physical.PType

object ScalaToRegionValue {
  def apply(region: Region, t: PType, a: Annotation): Long = {
    t.unstagedStoreJavaObject(a, region)
  }
}

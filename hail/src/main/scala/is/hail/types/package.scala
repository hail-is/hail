package is.hail

import is.hail.types.physical.PType
import is.hail.types.virtual.Type

package object types {
  def tcoerce[T <: Type](x: Type): T = x.asInstanceOf[T]

  def tcoerce[T <: PType](x: PType): T = x.asInstanceOf[T]

  def tcoerce[T <: BaseTypeWithRequiredness](x: BaseTypeWithRequiredness): T = x.asInstanceOf[T]
}

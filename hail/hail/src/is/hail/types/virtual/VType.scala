package is.hail.types.virtual

import is.hail.types.BaseType

import org.json4s.JValue

// types associated with BaseIRs
abstract class VType extends BaseType {
  def toJSON: JValue
}

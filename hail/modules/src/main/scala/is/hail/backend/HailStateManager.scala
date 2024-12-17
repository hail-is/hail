package is.hail.backend

import is.hail.variant.ReferenceGenome

case class HailStateManager(val referenceGenomes: Map[String, ReferenceGenome])
    extends Serializable {}

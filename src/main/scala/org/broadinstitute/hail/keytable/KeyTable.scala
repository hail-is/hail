package org.broadinstitute.hail.keytable

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.Type
import org.broadinstitute.hail.utils._


case class KeyTable (rdd: RDD[(Annotation, Annotation)], keySignature: Type, valueSignature: Type)

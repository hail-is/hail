package org.apache.spark

import scala.reflect._
import org.apache.spark.util._
import org.apache.spark.serializer._

object ExposedUtils {
  def clean[F <: AnyRef](
    sc: SparkContext,
    f: F, checkSerializable: Boolean = true
  ): F = sc.clean(f, checkSerializable)

  def clone[T: ClassTag](value: T, sc: SparkContext): T =
    clone(value, sc.env.serializer.newInstance())

  def clone[T: ClassTag](value: T, serializer: SerializerInstance): T =
    Utils.clone(value, serializer)
}

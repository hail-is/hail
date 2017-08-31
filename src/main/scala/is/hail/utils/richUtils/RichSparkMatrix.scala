package is.hail.utils.richUtils

import breeze.linalg.Matrix
import org.apache.spark.mllib.linalg.{Matrix => SparkMatrix}

class RichSparkMatrix(sparkMatrix: SparkMatrix) {
  /**
    * Converts to a Breeze matrix by reflecting into Spark and using their private method. This allows converting
    * from Spark to Breeze without calling "toArray" and doing an unnecessary matrix copy.
    */
  def asBreeze(): Matrix[Double] = {
    val breezeConverter = sparkMatrix.getClass.getMethod("asBreeze")
    breezeConverter.invoke(sparkMatrix).asInstanceOf[Matrix[Double]]
  }
}
package is.hail.utils.richUtils

import breeze.linalg.{Matrix => BreezeMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}

/**
  * Adds useful methods to Spark local Matrix interface.
  */
class RichSparkMatrix(sparkMatrix: Matrix) {
  /**
    * Converts to a Breeze matrix by reflecting into Spark and using their private method. This allows converting
    * from Spark to Breeze without calling "toArray" and doing an unnecessary matrix copy.
    */
  def asBreeze(): BreezeMatrix[Double] = {
    val breezeConverter = sparkMatrix.getClass.getMethod("asBreeze")
    breezeConverter.invoke(sparkMatrix).asInstanceOf[BreezeMatrix[Double]]
  }


}

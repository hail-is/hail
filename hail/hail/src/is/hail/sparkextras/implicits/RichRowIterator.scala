package is.hail.sparkextras.implicits

import is.hail.annotations.Region
import is.hail.types.physical.PStruct

import org.apache.spark.sql.Row

class RichRowIterator(val it: Iterator[Row]) extends AnyVal {
  def copyToRegion(region: Region, rowTyp: PStruct): Iterator[Long] =
    it.map(row => rowTyp.unstagedStoreJavaObject(null, row, region))
}

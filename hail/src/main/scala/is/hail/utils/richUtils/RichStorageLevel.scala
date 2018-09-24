package is.hail.utils.richUtils

import org.apache.spark.storage.StorageLevel

class RichStorageLevel(val sl: StorageLevel) extends AnyVal {

  def toReadableString(): String = {
    val rep = if (sl.replication == 1) "" else s"_${ sl.replication }"

    (sl.useDisk, sl.useMemory, sl.useOffHeap, sl.deserialized) match {
      case (false, false, false, false) => s"NONE$rep"
      case (true, false, false, false) => s"DISK_ONLY$rep"
      case (false, true, false, true) => s"MEMORY_ONLY$rep"
      case (false, true, false, false) => s"MEMORY_ONLY_SER$rep"
      case (true, true, false, true) => s"MEMORY_AND_DISK$rep"
      case (true, true, false, false) => s"MEMORY_AND_DISK_SER$rep"
      case (false, false, true, false) => s"OFF_HEAP$rep"
      case _ => sl.toString()
    }
  }
}
